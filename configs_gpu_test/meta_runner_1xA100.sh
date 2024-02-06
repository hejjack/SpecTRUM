#!/bin/bash
#PBS -q gpu@cerit-pbs.cerit-sc.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=4:ngpus=1:mem=50gb:cl_zia=True
#PBS -N GPU_test_ft_scratch_zia_1x

cd /storage/brno2/home/ahajek/Spektro/MassGenie/configs_gpu_test
source /storage/brno2/home/ahajek/miniconda3/bin/activate BARTtrainH100
echo $CONDA_PREFIX
./run_finetune_GPU_test_1xA100.sh

exit

##### 2x A40 ######
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=8:ngpus=2:mem=50gb:scratch_local=400mb:cl_galdor=True
#PBS -N GPU_test_ft_scratch__GALDOR


##### DGXko ########
# nnn #PBS -q gpu_dgx@meta-pbs.metacentrum.cz
# nnn #PBS -l walltime=24:0:0
# nnn #PBS -l select=1:ncpus=16:ngpus=4:mem=80gb
# nnn #PBS -N GPU_test_ft_scratch_capy
