#!/bin/bash
#PBS -q gpu_dgx@pbs-m1.metacentrum.cz
#PBS -l walltime=8:0:0
#PBS -l select=1:ncpus=16:ngpus=4:mem=80gb
#PBS -N GPU_test_ft_scratch_capy

cd /storage/brno2/home/ahajek/Spektro/MassGenie/config_runners
source /storage/brno2/home/ahajek/miniconda3/bin/activate BARTtrain
echo $CONDA_PREFIX
./run_finetune_from_scratch_GPU_test.sh