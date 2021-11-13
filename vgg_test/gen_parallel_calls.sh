#!/bin/bash
# Usage: ./gen_parallel_calls.sh
# Runs tests for varying parameters in batches of 8

declare -a optimizers=("SGD", "Adam", "RMSprop", "AdamW", "AggMo", "AdaBelief")
declare -a ablate_vals=("--ablate", "")
declare -a lr_vals=("sched", "0.1", "1e-2", "1e-3")
declare -a wd_vals=("1e-4", "0")

output_filename="run_parallel_calls.sh"
batch_size=6

#####################################################################################################

echo "#parallel calls" > $output_filename

CTR=0
for opt in "${optimizers[@]}"
do
for ablate in "${ablate_vals[@]}"
do
for lr in "${lr_vals[@]}"
do
for wd in "${wd_vals[@]}"
do
   let CTR=CTR+1
   echo "python3 vmain.py --optim $opt $ablate --lr $lr --wd $wd &" >> $output_filename
   if ! (($CTR % $batch_size)); then
	   echo "wait" >> $output_filename
   fi
done
done
done
done
