# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import time
import argparse
import os 

import numpy as np

import torch
from torch.autograd import Variable

from data import get_nli, get_batch, build_vocab
from mutils import write_to_csv, get_optimizer, construct_model_name
from models import NLINet, DebiasNet

parser = argparse.ArgumentParser(description='NLI training')
# losses.
parser.add_argument("--poe_alpha", type=float, default=1.0)
parser.add_argument("--gamma_focal", type=float, default=2.0)
parser.add_argument("--nonlinear_h_classifier", action="store_true", help="If specified uses a nonlinear classifier for h model.")
parser.add_argument("--use_early_stopping", action="store_true")
parser.add_argument("--rubi", action="store_true")
parser.add_argument("--poe_loss", action="store_true", help="Uses the product of the expert loss.")
parser.add_argument("--focal_loss", action="store_true", help="Uses the focal loss for classification,\
        where instead of the probabilities of the objects, we use the h only probabilities")

# paths
parser.add_argument("--outputfile", type=str, default="results.csv", help="writes the final results\
    in this file in a csv format.")

parser.add_argument("--dataset", type=str, default="SNLI", help="this will be set automatically.")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, nargs='+', default=['model.pickle'])
parser.add_argument("--word_emb_path", type=str, default="../../data/GloVe/glove.840B.300d.txt", help="word embedding file path")

# training
parser.add_argument('--h_loss_weight', type=float, default=1.0, help="defines the weight of the adversary loss.")
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", action="store_true", help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--version", type=int, default=2, help="Defines the version of the model.")
parser.add_argument("--encoder_type", type=str, default='InferSent', choices=['InferSent'], help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, unknowns = parser.parse_known_args()
if len(unknowns) != 0:
   raise AssertionError("There exists unknown parameters: ", unknowns)


all_datasets = {
            'SNLI': {'path': "../../data/datasets/SNLI", 'n_classes': 3},
            'SNLIHard': {'path': "../../data/datasets/SNLIHard", 'n_classes': 3},
            'MNLIMatched': {'path': "../../data/datasets/MNLIMatched/", 'n_classes': 3},
            'MNLIMismatched': {'path': "../../data/datasets/MNLIMismatched/", 'n_classes': 3},
            'MNLIMismatchedHardWithHardTest': {'path': "../../data/datasets/MNLIMismatchedHardWithHardTest/", 'n_classes':3},
            'MNLIMatchedHardWithHardTest': {'path': "../../data/datasets/MNLIMatchedHardWithHardTest/", 'n_classes':3},
            'JOCI': {'path': "../../data/datasets/JOCI", 'n_classes': 3},
            'SICK-E': {'path': "../../data/datasets/SICK-E", 'n_classes': 3},
            'AddOneRTE': {'path': "../../data/datasets/AddOneRTE", 'n_classes': 2},
            'DPR': {'path': "../../data/datasets/DPR", 'n_classes': 2},
            'FNPLUS': {'path': "../../data/datasets/FNPLUS", 'n_classes': 2},
            'SciTail': {'path': "../../data/datasets/SciTail", 'n_classes': 2},
            'SPRL': {'path': "../../data/datasets/SPRL", 'n_classes': 2},
            'MPE': {'path': "../../data/datasets/MPE", 'n_classes': 3},
            'QQP': {'path': "../../data/datasets/QQP", 'n_classes': 2},
            'GLUEDiagnostic': {'path': "../../data/datasets/GLUEDiagnostic", 'n_classes': 3},
}
params.nlipath = all_datasets[params.dataset]['path']
params.n_classes = all_datasets[params.dataset]['n_classes'] 
params.outputmodelname = construct_model_name(params, params.outputmodelname)

# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)

# this function clears the gradient of the given model.
def clear_gradients(model, name):
   for param in eval('model.'+name).parameters():
       if param.grad is not None:
          param.grad *= 0.0

"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
train, valid, test = get_nli(params.nlipath, params.n_classes)
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], params.word_emb_path)

for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])

"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,
    'version'        :  params.version        , 

}
nli_net = NLINet(config_nli_model)
print(nli_net)


config_debias_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,
    'nli_net'        :  nli_net               ,
    'version'        :  params.version        ,
    "poe_loss"       : params.poe_loss        ,
    "focal_loss"     : params.focal_loss      ,
    "h_loss_weight"  : params.h_loss_weight   ,
    "rubi"           : params.rubi            ,
    "nonlinear_h_classifier" : params.nonlinear_h_classifier,
    "gamma_focal" : params.gamma_focal,  
    "poe_alpha" : params.poe_alpha,
}

# model
encoder_types = ['InferSent']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
debias_net = DebiasNet(config_debias_model)
print(debias_net)


# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(debias_net.parameters(), **optim_params)
# cuda by default
debias_net.cuda()


"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    debias_net.train()

    all_costs = []

    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.

    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))
    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]


    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        k = s1_batch.size(1)  # actual batch size

        # model forward
        outputs = debias_net((s1_batch, s1_len), (s2_batch, s2_len), tgt_batch)

        pred = outputs['nli'].data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # define the losses here.
        all_costs.append(outputs['total_loss'].item())
        
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim
        # backward
        optimizer.zero_grad()
        # lets do the backward in the several steps.
        outputs['total_loss'].backward()


        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in debias_net.parameters():
            if p.requires_grad:
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm.cpu())

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            logs_outputs =  '{0} ; total loss {1} ; sentence/s {2} ;\
                            words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*correct.item()/(stidx+k), 2))
            logs.append(logs_outputs)
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct.item()/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    return train_acc


def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    debias_net.eval()

    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        outputs = debias_net((s1_batch, s1_len), (s2_batch, s2_len), tgt_batch)

        pred = outputs['nli'].data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * correct.item() / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(debias_net, os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr and params.use_early_stopping:
                    stop_training = True
            if 'adam' in params.optimizer and params.use_early_stopping:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc


"""
Train model on Natural Language Inference task
"""
epoch = 1

while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1

# Run best model on test set.
debias_net = torch.load(os.path.join(params.outputdir, params.outputmodelname))

scores = {}
print('\nTEST : Epoch {0}'.format(epoch))
scores['NLI_val'] = evaluate(1e6, 'valid', True)
scores['NLI_test'] = evaluate(0, 'test', True)

write_to_csv(scores, params, params.outputfile)
