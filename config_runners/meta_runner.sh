#!/bin/bash
#PBS -N BARTSpektro_experiment
#PBS -q gpu@cerit-pbs.cerit-sc.cz
#PBS -l select=1:ncpus=6:mem=70gb:scratch_local=1gb:ngpus=2:gpu_cap=cuda80:cluster=^fer
#PBS -l place=scatter
#PBS -m ae

cd /storage/brno2/home/ahajek/Spektro/MassGenie/config_runners
source /storage/brno2/home/ahajek/miniconda3/bin/activate BARTtrain
echo $CONDA_PREFIX
./run_finetune_4_8M_neims_gen.sh