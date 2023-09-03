#!/bin/bash
PRED_SCRIPT=~/Spektro/NEIMS/run_make_spectra_prediction.sh
plain_sdf=~/Spektro/MassGenie/tmp/tmp_for_NEIMS.sdf
enriched_sdf=~/Spektro/MassGenie/tmp/tmp_for_NEIMS_enriched.sdf


export PATH="/storage/brno2/home/ahajek/anaconda3/bin:$PATH"
. /storage/brno2/home/ahajek/.bashrc

conda activate NEIMSpy2

python --version

echo "##### Generating spectra #####"
. $PRED_SCRIPT $plain_sdf $enriched_sdf

conda deactivate