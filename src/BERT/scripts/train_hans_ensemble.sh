# Sets the paths to save the results and the outputs_dir
results_dir="results/bert/hans_ensemble_results/"
mkdir -p $results_dir
outputs_dir="/idiap/temp/rkarimi/debias/bert/hans_ensemble_experiments/"
mkdir -p $outputs_dir
# Sets the path to the downloaded BERT model and the cached_dir
cache_dir="/idiap/temp/rkarimi/cache_dir/"
bert_dir="/idiap/temp/rkarimi/pretrained_transformers/bert-base-uncased/"
seeds=(74 109 53)


# Train the POE ensemble model.
for seed in ${seeds[@]}; do
    python run_glue.py --cache_dir $cache_dir  --do_train --do_eval \
    --task_name mnli  --eval_task_names  mnli HANS MNLIMatchedHardWithHardTest \
    MNLIMismatchedHardWithHardTest --model_type bert --num_train_epochs 3  --lambda_h 1.0 \
    --model_name_or_path $bert_dir  --output_dir $outputs_dir/bert_poe_seed_$seed \
    --overwrite_output_dir --learning_rate 2e-5 --do_lower_case \
    --outputfile $results_dir/hans_bert_ensemble.csv  --poe_loss --nonlinear_h_classifier deep \
    --hans  --similarity mean max min second_min  --weighted_bias_only --binerize_eval \
    --seed $seed --hans_features  --length_features log-len-diff  --ensemble_training
done


# Train the DFL ensemble model.
for seed in ${seeds[@]}; do
    python run_glue.py --cache_dir $cache_dir  --do_train --do_eval \
    --task_name mnli  --eval_task_names  mnli HANS MNLIMatchedHardWithHardTest \
    MNLIMismatchedHardWithHardTest --model_type bert --num_train_epochs 3  --lambda_h 1.0 \
    --model_name_or_path $bert_dir --output_dir $outputs_dir/bert_focalloss_seed_$seed \
    --overwrite_output_dir --learning_rate 2e-5 --do_lower_case  \
    --outputfile  $results_dir/hans_bert_ensemble.csv  --focal_loss --nonlinear_h_classifier deep \
    --hans  --similarity mean max min second_min  --weighted_bias_only --binerize_eval --seed $seed \
    --hans_features --length_features log-len-diff --aggregate_ensemble mean  --ensemble_training
done
