#!/bin/bash
#PBS -q gpu_dgx@pbs-m1.metacentrum.cz
#PBS -l walltime=35:0:0
#PBS -l select=1:ncpus=7:ngpus=1:mem=50gb
#PBS -N run_pretrain_rassp1_neims1_nist01_224k

cd /storage/brno2/home/ahajek/Spektro/MassGenie/config_runners
source /storage/brno2/home/ahajek/miniconda3/bin/activate BARTtrainH100
echo $CONDA_PREFIX
./run_pretrain_1_1_01_224k.sh

exit

##### 1x A100 ######

##### 1x A40 ######
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=8:ngpus=1:mem=50gb:scratch_local=400mb:cl_galdor=True
#PBS -N debug_train__GALDOR


##### DGXko ########
#PBS -q gpu_dgx@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=7:ngpus=1:mem=50gb
#PBS -N run_pretrain_rassp1_neims1_nist01