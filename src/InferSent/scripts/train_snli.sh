#!/bin/bash

# Sets the paths to save the results and the trained models.
results_dir="results/snli_results/"
mkdir -p $results_dir
outputs_dir="/idiap/temp/rkarimi/debias/snli_experiments/"
mkdir -p $outputs_dir

#---------------------------
# Train the models on SNLI.
#--------------------------- 
# Train the baseline InferSent model.
python train_nli.py --h_loss_weight 0.0  --enc_lstm_dim 512 --version 2 \
--outputdir $outputs_dir --optimizer=sgd,lr=0.1 --nonlinear_fc --dataset SNLI \
--outputfile $results_dir/results.csv   --outputmodelname baseline


# Train RUBI model.
python train_nli.py --h_loss_weight 1.0  --enc_lstm_dim 512 --version 2 --dataset SNLI \
--optimizer=sgd,lr=0.1 --rubi --nonlinear_fc --outputfile $results_dir/results.csv \
--outputmodelname RUBI --outputdir $outputs_dir

# Train DFL model.
python train_nli.py --h_loss_weight 1.0 --enc_lstm_dim 512 --version 2 \
--outputdir $outputs_dir --optimizer=sgd,lr=0.1 --focal_loss --nonlinear_fc \
--dataset SNLI --outputfile $results_dir/results.csv --outputmodelname DFL


# Train POE model.
python train_nli.py --h_loss_weight 1.0  --enc_lstm_dim 512 --version 2 \
--outputdir $outputs_dir --optimizer=sgd,lr=0.1 --poe_loss  --nonlinear_fc \
--dataset SNLI --outputfile $results_dir/results.csv --outputmodelname POE


#---------------------------------------------------
# Evaluation of the trained models on the hard set. 
#---------------------------------------------------
# Evaluates the trained baseline model on SNLIHard set.
python train_nli.py --h_loss_weight 0.0  --enc_lstm_dim 512 --version 2 \
--outputdir $outputs_dir --optimizer=sgd,lr=0.1 --nonlinear_fc --dataset SNLIHard \
--outputfile $results_dir/results_hard.csv --outputmodelname baseline --n_epochs 0


# Evaluates the trainred RUBI model on the SNLIHard set.
python train_nli.py --h_loss_weight 1.0  --enc_lstm_dim 512 --version 2 \
--dataset SNLIHard --optimizer=sgd,lr=0.1 --rubi --nonlinear_fc \
--outputfile $results_dir/results_hard.csv   --outputmodelname RUBI \
--outputdir $outputs_dir --n_epochs 0 


# Evaluates the trained DFL model on the SNLIHard set.
python train_nli.py --h_loss_weight 1.0 --enc_lstm_dim 512 --version 2 \
--outputdir $outputs_dir --optimizer=sgd,lr=0.1 --focal_loss --nonlinear_fc \
--dataset SNLIHard --outputfile $results_dir/results_hard.csv --outputmodelname DFL \
--n_epochs 0 


# Evaluates the trained PoE model on the SNLIHard set.
python train_nli.py --h_loss_weight 1.0  --enc_lstm_dim 512 --version 2 \
--outputdir $outputs_dir --optimizer=sgd,lr=0.1 --poe_loss  --nonlinear_fc \
--dataset SNLIHard --outputfile $results_dir/results_hard.csv --outputmodelname POE \
--n_epochs 0
