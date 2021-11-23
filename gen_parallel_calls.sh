#!/bin/bash
# Usage: ./gen_parallel_calls.sh
# Runs tests for varying parameters in batches of 8

declare -a optimizers=("SGD" "Adam" "RMSprop")
declare -a ablate_vals=("--ablate" "")
# declare -a lr_vals=("sched" "0.1" "1e-2" "1e-3")
declare -a lr_vals=("0.1" "1e-2" "1e-3" "1e-4")
declare -a wd_vals=("1e-4" "0")
declare -a mom_vals=("0" "0.9")

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
   if [ $opt == "SGD" ];
   then
     for m in "${mom_vals[@]}"
     do
       let CTR=CTR+1
       echo "python3 vmain.py --optim $opt $ablate --lr $lr --wd $wd --mom $m &" >> $output_filename
       if ! (($CTR % $batch_size)); then
        echo "wait" >> $output_filename
       fi
     done
   elif [ $opt == "RMSprop" ];
   then
     for m in "${mom_vals[@]}"
     do
       let CTR=CTR+1
       echo "python3 vmain.py --optim $opt $ablate --lr $lr --wd $wd --mom $m &" >> $output_filename
       if ! (($CTR % $batch_size)); then
        echo "wait" >> $output_filename
       fi
     done
   else
     let CTR=CTR+1
     echo "python3 vmain.py --optim $opt $ablate --lr $lr --wd $wd &" >> $output_filename
     if ! (($CTR % $batch_size)); then
      echo "wait" >> $output_filename
     fi
   fi
done
done
done
done
