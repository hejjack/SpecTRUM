#!/bin/bash
#PBS -N 1101_runner
#PBS -q gpu@cerit-pbs.cerit-sc.cz
#PBS -l select=1:ncpus=5:mem=70gb:scratch_local=1gb:ngpus=1:gpu_cap=cuda80:cluster=^fer
#PBS -l place=scatter
#PBS -m ae

cd /storage/brno2/home/ahajek/Spektro/MassGenie/config_runners
source /storage/brno2/home/ahajek/miniconda3/bin/activate BARTtrain
echo $CONDA_PREFIX
./run_pretrain_1_1_01.sh