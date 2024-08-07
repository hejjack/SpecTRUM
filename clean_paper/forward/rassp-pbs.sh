#!/bin/bash

usage() {
	echo usage: $0 [-m model] [-e meta] [-i image] [-w workdir] smiles >&2
	exit 2
}

unset model meta image dir

while getopts m:e:i:w: opt; do case "$opt" in
	m) model="$OPTARG" ;;
	e) meta="$OPTARG" ;;
	i) image="$OPTARG" ;;	# must be absolute path
	w) dir="$OPTARG" ;;

	*) usage ;;
esac; done
shift $((OPTIND-1))

[ -z "$1" ] && usage

: ${SCRATCHDIR:=$TMPDIR}
: ${SCRATCHDIR:=/tmp}

cd "${dir:=$PBS_O_WORKDIR}" || exit 1
cp "$1" $SCRATCHDIR


if [ -z "$image" ]; then
	singularity pull $SCRATCHDIR/rassp.sif docker://ljocha/rassp:nvidia-2023-6
	image=$SCRATCHDIR/rassp.sif
fi

if [ -n "$model" ]; then
	cp "$model" $SCRATCHDIR/formulanet.model
	cp "$meta" $SCRATCHDIR/formulanet.meta
else
	curl -k -o $SCRATCHDIR/formulanet.model 'https://people.cs.uchicago.edu/~ericj/rassp/formulanet_best_candidate_pcsim_pretrain.nist-fromscratch-3x9x128.35790555.00000740.model' 
	curl -k -o $SCRATCHDIR/formulanet.meta 'https://people.cs.uchicago.edu/~ericj/rassp/formulanet_best_candidate_pcsim_pretrain.nist-fromscratch-3x9x128.35790555.meta' 
fi

cp $(dirname $0)/rassp-predict.py $SCRATCHDIR

cd $SCRATCHDIR || exit 1

unset nv
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
	CUDA_VISIBLE_DEVICES=$(nvidia-smi -L | grep $CUDA_VISIBLE_DEVICES | sed 's/^GPU \([0-9]*\):.*/\1/')
	nv=--nv
fi

chunk=$1
split -d -l 1000 ${chunk} ${chunk}_

for c in ${chunk}_900*; do
	mv $c $(echo $c | sed 's/900\(.\)/9\1/')
done

rm -f *.jsonl
for c in ${chunk}_*; do
	singularity exec $nv -B $PWD:/work --pwd /work "$image" /opt/nvidia/nvidia_entrypoint.sh python3 rassp-predict.py -m formulanet.model -e formulanet.meta -w ${PBS_NCPUS:-1} -s "$c" -o "$c.jsonl"
done

cat *.jsonl >"$dir/$1.jsonl"

# rm -rf *



