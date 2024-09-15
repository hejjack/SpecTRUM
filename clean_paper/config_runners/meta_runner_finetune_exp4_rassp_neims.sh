#!/bin/bash
#PBS -q gpu_dgx@pbs-m1.metacentrum.cz
#PBS -l walltime=20:0:0
#PBS -l select=1:ncpus=7:ngpus=1:mem=50gb
#PBS -N run_finetune_exp4_rassp_neims

cd /storage/brno2/home/ahajek/Spektro/MassGenie/clean_paper
source /storage/brno2/home/ahajek/miniconda3/bin/activate BARTtrainH100
echo $CONDA_PREFIX
./config_runners/run_finetune_exp4_rassp_neims.sh

exit

##### 1x A100 ######
#PBS -q gpu@cerit-pbs.cerit-sc.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=4:ngpus=1:mem=50gb:cl_zia=True
#PBS -N run_pretrain_rassp1_neims1

##### 1x A40 ######
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=8:ngpus=1:mem=50gb:scratch_local=400mb:cl_galdor=True
#PBS -N debug_train__GALDOR


##### DGXko ########