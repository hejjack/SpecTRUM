#!/bin/bash

if [ -z "$1" ]; then
	usage: $0 file1.sdf file2.sdf ... >&2; exit 1
fi

: ${SCRATCHDIR:=$TMPDIR}
: ${SCRATCHDIR:=/tmp}

cd "${dir:=$PBS_O_WORKDIR}" || exit 1
cd $SCRATCHDIR || exit 1

image=$dir/neims.sif
if [ ! -f $image ]; then
	singularity pull neims.sif docker://cerit.io/ljocha/neims
	image=$SCRATCHDIR/neims.sif
fi

unset nv
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
	CUDA_VISIBLE_DEVICES=$(nvidia-smi -L | grep $CUDA_VISIBLE_DEVICES | sed 's/^GPU \([0-9]*\):.*/\1/')
	nv=--nv
fi

md5sum -c - <<EOF
1c56b3f7545aca54398fbe0f8af0fb9e  massspec_weights.zip
EOF

if [ $? != 0 ]; then
	curl -o massspec_weights.zip https://storage.googleapis.com/deep-molecular-massspec/massspec_weights/massspec_weights.zip
fi

rm -rf massspec_weights
unzip massspec_weights.zip

cd $dir
mkdir $SCRATCHDIR/sdf
cp "$@" $SCRATCHDIR/sdf

cd $SCRATCHDIR/sdf
for f in *; do
	ofn=$(basename $f .sdf)-out.sdf
	singularity exec $nv -B $SCRATCHDIR:/work --pwd /work "$image" python3 /opt/neims/make_spectra_prediction.py \
		--input_file "sdf/$f" \
	       	--output_file "sdf/$ofn" \
		--weights_dir /work/massspec_weights
done

cp *-out.sdf $dir
