seeds=(186 44 66 53)
results_dir="results/bert/hans_results/"
mkdir -p  $results_dir
outputs_dir="/idiap/temp/rkarimi/debias/bert/hans_experiments/"
mkdir -p $outputs_dir 
cache_dir="/idiap/temp/rkarimi/cache_dir/"
bert_dir="/idiap/temp/rkarimi/pretrained_transformers/bert-base-uncased/"

# Due to the reported various on this dataset, we report the 
# average results across 4 seeds.
# Train the baseline model.
for seed in ${seeds[@]}; do 
    python run_glue.py --cache_dir $cache_dir --do_train --do_eval  \
    --task_name mnli --eval_task_names   mnli HANS  --model_type bert \
    --num_train_epochs 3  --lambda_h 0.0   --model_name_or_path $bert_dir \
    --output_dir $outputs_dir/bert_baseline_seed_$seed --overwrite_output_dir \
    --learning_rate 2e-5 --do_lower_case  --outputfile $results_dir/results.csv \
    --binerize_eval  --seed $seed 
done 


# Train the RUBI model.
for seed in ${seeds[@]}; do
    python run_glue.py --cache_dir $cache_dir --do_train --do_eval  \
    --task_name mnli --eval_task_names   mnli HANS  --model_type bert --num_train_epochs 3 \
    --lambda_h 1.0   --model_name_or_path $bert_dir --output_dir  $outputs_dir/bert_rubi_seed_$seed \
    --overwrite_output_dir --rubi --learning_rate 2e-5 --do_lower_case --outputfile $results_dir/results.csv \
    --nonlinear_h_classifier deep  --hans  --similarity mean max min --weighted_bias_only --binerize_eval \
    --seed $seed --hans_features
done


# Train the DFL model.
for seed in ${seeds[@]}; do
    python run_glue.py --cache_dir $cache_dir --do_train --do_eval  --task_name mnli \
    --eval_task_names  mnli HANS  --model_type bert --num_train_epochs 3  --lambda_h 1.0 \
    --model_name_or_path $bert_dir --output_dir $outputs_dir/bert_focalloss_seed_$seed \
    --overwrite_output_dir --learning_rate 2e-5 --do_lower_case --outputfile $results_dir/results.csv \
    --focal_loss   --nonlinear_h_classifier deep  --hans  --similarity mean max min \
    --weighted_bias_only  --binerize_eval  --seed $seed --hans_features
done 


# Train the POE model.
for seed in ${seeds[@]}; do
    python run_glue.py --cache_dir $cache_dir --do_train --do_eval --task_name mnli \
    --eval_task_names mnli HANS --model_type bert --num_train_epochs 3  --lambda_h 1.0 \
    --model_name_or_path $bert_dir --output_dir $outputs_dir/bert_poeloss_seed_$seed \
    --overwrite_output_dir --learning_rate 2e-5 --do_lower_case  --outputfile $results_dir/results.csv \
    --poe_loss   --nonlinear_h_classifier deep --hans  --similarity mean max min \
    --weighted_bias_only  --binerize_eval  --seed $seed --hans_features
done
