SING_IMAGE=/storage/brno6/home/ahajek/singularity/ngc_image/Pytorch-21.SIF
SING_IMAGE=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:22.04.SIF 

HOMEDIR=/scratch.ssd/ahajek  # substitute username and path to to your real username and path
IMAGE_BASE=`basename $SING_IMAGE`
export PYTHONUSERBASE=$HOMEDIR/.local-${IMAGE_BASE}

mkdir -p ${PYTHONUSERBASE}/lib/python3.6/site-packages 

#set SINGULARITY variables for runtime data
export SINGULARITY_CACHEDIR=$HOMEDIR
export SINGULARITY_LOCALCACHEDIR=$SCRATCHDIR
export SINGULARITY_TMPDIR=$SCRATCHDIR
export SINGULARITYENV_PREPEND_PATH=$PYTHONUSERBASE/bin:$PATH

cd /scratch.ssd/ahajek/??????

# jupyter nbconvert --to notebook --execute my_train_noconotr3in3.ipynb --inplace
singularity exec --nv -H $HOMEDIR --bind /tmp --bind /scratch.ssd --env KRB5CCNAME=$KRB5CCNAME $SING_IMAGE jupyter nbconvert --to notebook --execute train_bart_bigdata.ipynb --inplace
# runipy -o my_train_noconotr3in3.ipynb
# singularity exec --nv -B /tmp,$SCRATCHDIR --env KRB5CCNAME=$KRB5CCNAME -H /home/ahajek $IMAGE_PATH runipy -o my_train_noconotr3in1.ipynb


