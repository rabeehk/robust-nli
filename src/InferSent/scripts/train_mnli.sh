#!/bin/bash

# Sets the path to save the results and the trained models.
results_dir="results/mnli_results/"
mkdir -p $results_dir
outputs_dir="/idiap/temp/rkarimi/debias/mnli_experiments/"
mkdir -p $outputs_dir

#------------------------------------------------
# Train the models on MNLIMatched/MNLIMismatched.
#------------------------------------------------
# Train the baseline InferSent model.
for data in MNLIMatched MNLIMismatched
do
    python train_nli.py --h_loss_weight 0.0  --enc_lstm_dim 512 --version 2 \
    --outputdir $outputs_dir --optimizer=sgd,lr=0.1 --nonlinear_fc --dataset $data \
    --outputfile $results_dir/results.csv   --outputmodelname baseline
done


# Train RUBI model.
for data in MNLIMatched MNLIMismatched
do    
    python train_nli.py --h_loss_weight 1.0  --enc_lstm_dim 512 --version 2 \
    --outputdir $outputs_dir --dataset $data --optimizer=sgd,lr=0.1 --rubi \
    --nonlinear_fc  --outputfile $results_dir/results.csv --outputmodelname RUBI
done


# Train DFL model.
for data in MNLIMatched MNLIMismatched
do
    python train_nli.py --h_loss_weight 1.0 --enc_lstm_dim 512 --version 2 \
    --outputdir $outputs_dir  --optimizer=sgd,lr=0.1 --focal_loss --nonlinear_fc \
    --dataset $data --outputfile $results_dir/results.csv --outputmodelname DFL
done


# Train POE model.
for data in MNLIMatched MNLIMismatched
do
    python train_nli.py --h_loss_weight 1.0  --enc_lstm_dim 512 --version 2 \
    --outputdir $outputs_dir  --optimizer=sgd,lr=0.1 --poe_loss  --nonlinear_fc \
    --dataset $data --outputfile $results_dir/results.csv --outputmodelname POE
done

#------------------------------------------------
# Evaluates the trained models on the hard sets.
#------------------------------------------------
# Evaluates the baseline InferSent model.
for data in MNLIMismatchedHardWithHardTest MNLIMatchedHardWithHardTest
do 
    python train_nli.py --h_loss_weight 0.0  --enc_lstm_dim 512 --version 2 \
    --outputdir $outputs_dir --optimizer=sgd,lr=0.1 --nonlinear_fc --dataset $data \
    --outputfile $results_dir/results_hard.csv   --outputmodelname baseline --n_epochs 0
done 


# Evaluates the RUBI model.
for data in MNLIMismatchedHardWithHardTest MNLIMatchedHardWithHardTest
do
    python train_nli.py --h_loss_weight 1.0  --enc_lstm_dim 512 --version 2 \
    --outputdir $outputs_dir --dataset $data --optimizer=sgd,lr=0.1 --rubi --nonlinear_fc \
    --outputfile $results_dir/results_hard.csv --outputmodelname RUBI --n_epochs 0
done


# Evlauates the DFL model.
for data in MNLIMismatchedHardWithHardTest MNLIMatchedHardWithHardTest
do 
    python train_nli.py --h_loss_weight 1.0 --enc_lstm_dim 512 --version 2 \
    --outputdir $outputs_dir  --optimizer=sgd,lr=0.1 --focal_loss --nonlinear_fc \
    --dataset $data --outputfile $results_dir/results_hard.csv --outputmodelname DFL --n_epochs 0
done


# Evaluates the POE model.
for data in MNLIMismatchedHardWithHardTest MNLIMatchedHardWithHardTest
do
    python train_nli.py --h_loss_weight 1.0  --enc_lstm_dim 512 --version 2 \
    --outputdir $outputs_dir --optimizer=sgd,lr=0.1 --poe_loss  --nonlinear_fc \
    --dataset $data --outputfile $results_dir/results_hard.csv --outputmodelname POE --n_epochs 0
done
