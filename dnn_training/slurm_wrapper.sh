#!/bin/bash
 
##Request 1 gpu:
#SBATCH --gres=gpu:teslak80:1
#SBATCH -p gpu
#SBATCH --mem-per-cpu 12G
#SBATCH -t 53:59:59

## system message output file
###SBATCH -o /scratch/work/rkarhila/another_rnn_phone_classifier/triton_logs/.gpu_%j.out

## name of your job
#SBATCH -J GPU_tf_train

# Run in anaconda virtual env that includes Tensor Flow and Keras:
module load anaconda3 CUDA/7.5.18 cudnn/4
source activate tensorflow3
 
## run my GPU accelerated executable with --gres
## srun --gres=gpu:teslak80:1 

python rnn_training_37melbin_4x1000_triton-2-two_languages_and_map_threshold0.3_balance0.tmp.py

#python rnn_training_37melbin_4x1000_triton-2-only-finnish.py
#python rnn_training_37melbin_4x1000_triton-2-only-english.py
