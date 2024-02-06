#!/bin/bash
#PBS -q gpu_dgx@meta-pbs.metacentrum.cz
#PBS -l walltime=18:0:0
#PBS -l select=1:ncpus=8:ngpus=2:mem=60gb
#PBS -N GPU_test_ft_scratch_large_capy_2x

cd /storage/brno2/home/ahajek/Spektro/MassGenie/configs_gpu_test
source /storage/brno2/home/ahajek/miniconda3/bin/activate BARTtrainH100
echo $CONDA_PREFIX
./run_finetune_GPU_test_2xH100_large.sh
