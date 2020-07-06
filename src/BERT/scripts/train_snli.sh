#!/bin/bash

# Sets the paths to save the results and the trained models.
results_dir="results/bert/snli_results/"
mkdir -p $results_dir
outputs_dir="/idiap/temp/rkarimi/debias/bert/snli_experiments/"
mkdir -p $outputs_dir
# Sets the paths to the downloaded BERT model and the cache dir.
cache_dir="/idiap/temp/rkarimi/cache_dir/"
bert_dir="/idiap/temp/rkarimi/pretrained_transformers/bert-base-uncased/"


# Train the baseline model.
python run_glue.py --cache_dir $cache_dir  --do_train --do_eval \
--task_name snlihard  --eval_task_names snli snlihard --model_type bert \
--num_train_epochs 3 --lambda_h 0.0 --model_name_or_path $bert_dir \
--output_dir $outputs_dir/bert_baseline  --overwrite_output_dir  \
--learning_rate 2e-5 --do_lower_case  --outputfile $results_dir/results.csv \
--binerize_eval --gamma_focal 0.0

# Train the RUBI model.
python run_glue.py --cache_dir $cache_dir --do_train --do_eval \
--task_name snlihard  --eval_task_names  snli snlihard --model_type bert \
--num_train_epochs 3 --lambda_h 1.0 --model_name_or_path $bert_dir  \
--output_dir $outputs_dir/bert_rubi --overwrite_output_dir --rubi  \
--learning_rate 2e-5 --do_lower_case  --outputfile $results_dir/results.csv \
--nonlinear_h_classifier deep  --binerize_eval


# Train the DFL model.
python run_glue.py --cache_dir $cache_dir  --do_eval --do_train \
--task_name snlihard  --eval_task_names  snli snlihard --model_type bert \
--num_train_epochs 3 --lambda_h 1.0  --model_name_or_path $bert_dir \
--output_dir  $outputs_dir/bert_focal  --overwrite_output_dir --learning_rate 2e-5 \
--do_lower_case  --outputfile $results_dir/results.csv --focal_loss \
--nonlinear_h_classifier deep  --binerize_eval


# Train the POE model.
python run_glue.py --cache_dir $cache_dir  --do_eval --do_train \
--task_name snlihard  --eval_task_names  snli snlihard --model_type bert \
--num_train_epochs 3  --lambda_h 1.0  --model_name_or_path  $bert_dir \
--output_dir $outputs_dir/bert_poe --overwrite_output_dir \
--learning_rate 2e-5 --do_lower_case  --outputfile $results_dir/results.csv \
--poe_loss --nonlinear_h_classifier deep  --binerize_eval
