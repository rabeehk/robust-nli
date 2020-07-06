""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import logging
import os
import random
from utils_glue import GLUE_TASKS_NUM_LABELS
from eval_utils import load_and_cache_examples, evaluate, get_parser
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler)
from tqdm import tqdm, trange
from mutils import write_to_csv

from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer, 
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from utils_bert import BertDebiasForSequenceClassification
from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (compute_metrics, convert_examples_to_features,
                        processors)
from eval_utils import task_to_data_dir, nli_task_names, actual_task_names, ALL_MODELS


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)


from eval_utils import MODEL_CLASSES
from eval_utils import do_evaluate


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def save_model(args, global_step, model, logger):
    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    logger.info("Saving model checkpoint to %s", output_dir)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.hans:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3],
                          'h_ids': batch[4],
                          'h_attention_mask': batch[5],
                          'p_ids': batch[6],
                          'p_attention_mask': batch[7],
                          'have_overlap': batch[8],
                          'overlap_rate': batch[9],
                          'subsequence': batch[10],
                          'constituent': batch[11]
                }
            elif args.rubi or args.hypothesis_only or args.focal_loss or args.poe_loss:
                inputs = {'input_ids':        batch[0],
                          'attention_mask':   batch[1],
                          'token_type_ids':   batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':           batch[3],
                          'h_ids':            batch[4],
                          'h_attention_mask': batch[5]}
            else:
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3]}


            outputs = model(**inputs)
            loss = outputs["bert"][0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    save_model(args, global_step, model, logger)
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    #tb_writer.close()
    return global_step, tr_loss / global_step


def main():
    parser = get_parser()
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))


    # add all variations of hans automatically
    if "HANS" in args.eval_task_names:
        hans_variations = ["HANS-const", "HANS-lex", "HANS-sub"]
        for variation in hans_variations:
            if variation not in args.eval_task_names:
               args.eval_task_names.append(variation)


    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda")
    args.device = device

    # All of these tasks use the NliProcessor
    args.actual_task_names = actual_task_names

    # By default we evaluate on the task itself.
    if len(args.eval_task_names) == 0:
        args.eval_task_names = [args.task_name]
    if "all" in args.eval_task_names:
        args.eval_task_names = args.eval_task_names + nli_task_names + ["snli", "mnli"]
        args.eval_task_names.remove("all")
    print(args.eval_task_names)

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    -1, device, 1, bool(False), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()

    if args.task_name.startswith("fever"):
       processor = processors["fever"]()
    elif args.task_name in nli_task_names:
       processor = processors["nli"](task_to_data_dir[args.task_name])
    elif args.task_name in ["mnli"]:
        processor = processors["mnli"](hans=args.hans)
    elif args.task_name.startswith("HANS"):
        processor = processors["hans"]()
    elif args.task_name in args.actual_task_names:
        processor = processors[args.task_name]()
    else:
        raise ValueError("Task not found: %s" % (args.task_name))
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    # Adds rubi parameters here.
    config.rubi = args.rubi
    config.hans = args.hans
    config.nonlinear_h_classifier = args.nonlinear_h_classifier
    config.hypothesis_only = args.hypothesis_only 
    config.lambda_h = args.lambda_h
    config.focal_loss = args.focal_loss
    config.poe_loss = args.poe_loss 
    config.similarity = args.similarity
    config.gamma_focal = args.gamma_focal
    config.weighted_bias_only = args.weighted_bias_only
    config.length_features = args.length_features 
    config.hans_features=args.hans_features
    config.hans_only = args.hans_only
    config.ensemble_training = args.ensemble_training
    config.aggregate_ensemble = args.aggregate_ensemble
    config.poe_alpha = args.poe_alpha

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    model.to(args.device)


    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset, _, _ = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        print("model is saved in ", os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        result, _ = do_evaluate(args, args.output_dir, tokenizer, model, config)
        for r in result:
            results.update(r)

    # saves the results.
    print(results)
    if args.outputfile is not None:
        write_to_csv(results, args, args.outputfile)
    return results


if __name__ == "__main__":
    main()

