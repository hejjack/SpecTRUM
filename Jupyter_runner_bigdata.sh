TOKEN=e53e6964fa96d16903e8dfdc9be9fe95e4a02fe3892ea2dc
SING_IMAGE=/storage/brno2/home/ahajek/singularity/ngc_image/Pytorch-21.SIF
HOMEDIR=/storage/brno2/home/ahajek
PORT="8765"
IMAGE_BASE=`basename $SING_IMAGE`
export PYTHONUSERBASE=$HOMEDIR/.local-${IMAGE_BASE}

mkdir -p ${PYTHONUSERBASE}/lib/python3.6/site-packages 

#find nearest free port to listen
isfree=$(netstat -taln | grep $PORT)
while [[ -n "$isfree" ]]; do
    PORT=$[PORT+1]
    isfree=$(netstat -taln | grep $PORT)
done

echo http://$HOSTNAME:$PORT

singularity exec --nv -H $HOMEDIR \
                 --bind /storage \
                 --bind /scratch.ssd \
                 $SING_IMAGE jupyter-lab --port $PORT --NotebookApp.token=$TOKEN \

