#!/bin/bash
#PBS -q gpu_dgx@pbs-m1.metacentrum.cz
#PBS -l walltime=18:0:0
#PBS -l select=1:ncpus=16:ngpus=4:mem=60gb
#PBS -N GCMS_large_capy_4x

cd /storage/brno2/home/ahajek/Spektro/MassGenie/configs_gpu_test
source /storage/brno2/home/ahajek/miniconda3/bin/activate BARTtrainH100
echo $CONDA_PREFIX
./run_finetune_GPU_test_4xH100_large.sh
