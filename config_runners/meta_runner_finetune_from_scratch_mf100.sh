#!/bin/bash
#PBS -q gpu_dgx@meta-pbs.metacentrum.cz
#PBS -l walltime=20:0:0
#PBS -l select=1:ncpus=7:ngpus=1:mem=50gb
#PBS -N run_finetune_from_scratch_mf100

cd /storage/brno2/home/ahajek/Spektro/MassGenie/config_runners
source /storage/brno2/home/ahajek/miniconda3/bin/activate BARTtrainH100
echo $CONDA_PREFIX
./run_finetune_from_scratch_mf100.sh

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