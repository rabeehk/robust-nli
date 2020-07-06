# Sets the path to save the results and and the trained models.
results_dir="results/bert/fever_results/"
mkdir -p $results_dir
outputs_dir="/idiap/temp/rkarimi/debias/bert/fever_experiments/"
mkdir -p $outputs_dir
# Sets the path to the downloaded BERT model and the cache directory.
cache_dir="/idiap/temp/rkarimi/cache_dir/"
bert_dir="/idiap/temp/rkarimi/pretrained_transformers/bert-base-uncased/"


# Train the baseline model.
python run_glue.py --cache_dir $cache_dir --do_train --do_eval \
--max_seq_length 128 --per_gpu_train_batch_size 32  --task_name fever \
--eval_task_names fever  fever-symmetric-generated --model_type bert \
--num_train_epochs 3 --lambda_h 0.0 --model_name_or_path $bert_dir \
--output_dir $outputs_dir/baseline_fever  --learning_rate 2e-5 --do_lower_case \
--outputfile $results_dir/results.csv  --overwrite_output_dir


# Train the RUBI model.
python run_glue.py --cache_dir $cache_dir --do_train --do_eval \
--max_seq_length 128 --per_gpu_train_batch_size 32  --task_name fever \
--eval_task_names fever  fever-symmetric-generated --model_type bert \
--num_train_epochs 3 --lambda_h 1.0 --model_name_or_path  $bert_dir \
--output_dir $outputs_dir/rubi  --learning_rate 2e-5 --do_lower_case \
--outputfile $results_dir/results.csv  --overwrite_output_dir --rubi \
--nonlinear_h_classifier deep  --rubi_text a


# Train the DFL model.
python run_glue.py --cache_dir $cache_dir  --do_train --do_eval \
--max_seq_length 128 --per_gpu_train_batch_size 32  --task_name fever \
--eval_task_names fever  fever-symmetric-generated --model_type bert \
--num_train_epochs 3   --lambda_h 1.0    --model_name_or_path  $bert_dir \
--output_dir $outputs_dir/DFL  --learning_rate 2e-5 --do_lower_case  \
--outputfile $results_dir/results.csv --overwrite_output_dir --focal_loss \
--nonlinear_h_classifier deep  --rubi_text a


# Train the POE model.
python run_glue.py --cache_dir $cache_dir --do_train --do_eval \
--max_seq_length 128  --per_gpu_train_batch_size 32  --task_name fever \
--eval_task_names fever  fever-symmetric-generated --model_type bert \
--num_train_epochs 3 --lambda_h 1.0 --model_name_or_path $bert_dir \
--output_dir $outputs_dir/PoE --learning_rate 2e-5 --do_lower_case \
--outputfile  $results_dir/results.csv  --overwrite_output_dir --poe_loss \
--nonlinear_h_classifier deep  --rubi_text a
