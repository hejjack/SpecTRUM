#!/bin/bash
# STATS:
# 2M
# train len: 585996
# test len: 73249
# valid len: 73250 

# 8M
# train len: 4641078
# test len: 
# valid len:

# 1M 
# train len: 354767
# test len: 44346
# valid len: 44346
# setting Si as 580 (max token ID => 581 possible tokens)

# 1K
# train len: 469
# test len: 58
# valid len: 59
#### setting Br as 578 (max token ID => 579 possible tokens)

SPECTRO_DIR=/storage/projects/msml/mg_neims_branch
SPECTRO_DIR=/storage/brno2/home/ahajek/Spektro #??? (pridano 5/9/2022) 


export INPUT=NIST20_only     #12M_derivatized
SMILES_DATA_PATH=$SPECTRO_DIR/MassGenie/data/trial_set/$INPUT.smi
export INPUT=NEIMS_test_deriv
SMILES_DATA_PATH=$SPECTRO_DIR/NEIMS/training_splits/test_set_smiles_deriv.smi
TOKENIZER=_bbpe_1M

after_phase1_pickle=$SPECTRO_DIR/MassGenie/data/trial_set/tmp/${INPUT}_df_after_phase1.pkl
only_canon_smiles=$SPECTRO_DIR/MassGenie/data/trial_set/tmp/${INPUT}_only_canon_smiles.pkl
destereo_smiles=$SPECTRO_DIR/MassGenie/data/trial_set/tmp/${INPUT}_destereo_smiles.pkl
after_phase2_pickle=$SPECTRO_DIR/MassGenie/data/trial_set/tmp/${INPUT}_df_after_phase2.pkl
plain_sdf=${SPECTRO_DIR}/MassGenie/data/trial_set/tmp/${INPUT}_plain.sdf
enriched_sdf=${SPECTRO_DIR}/MassGenie/data/trial_set/tmp/${INPUT}_enriched_smiles.sdf
after_phase3_pickle=$SPECTRO_DIR/MassGenie/data/trial_set/tmp/${INPUT}_df_after_phase3.pkl
save_prepared_path=$SPECTRO_DIR/MassGenie/data/trial_set/${INPUT}${TOKENIZER}_bart_prepared_data.pkl

# NEJAK VYMYSLET MAZANI SOUBORU, AT TO NEMA TERABYTE


{
################### PREPROCESS 1 (CANONICALIZATION)
ARGS1="--data-path $SMILES_DATA_PATH \
       --save-pickle-path $after_phase1_pickle
       --save-canon-path $only_canon_smiles
"
# echo PHASE1 SKIPPED
python data_preprocess1.py $ARGS1

} && {
################### DESTEREOCHEMICALIZATION
export PATH="/storage/brno2/home/ahajek/anaconda3/bin:$PATH"
. /storage/brno2/home/ahajek/.bashrc

# echo DESTEREO SKIPPED
conda activate NEIMSpy2 && obabel -i smi $only_canon_smiles -o can -xi > $destereo_smiles && conda deactivate

} && {
################### PREPROCESS 2 (SDF PREPARATION)
ARGS2="--df-path $after_phase1_pickle 
       --destereo-path $destereo_smiles
       --save-pickle-path $after_phase2_pickle
       --save-plain-sdf-path $plain_sdf
       --max-smiles-len 100
       "
#       --load-pickle-path $after_phase2_pickle

# --load-pickle-path $ ???

# echo conda env is: $CONDA_DEFAULT_ENV

# echo PHASE2 SKIPPED
python data_preprocess2.py $ARGS2

} && {
################### NEIMS SPECTRA GENERATION
PRED_SCRIPT=$SPECTRO_DIR/NEIMS/run_make_spectra_prediction.sh

conda activate NEIMSpy2

echo NEIMS GENERATING SKIPPED
echo "##### Generating spectra #####"
. $PRED_SCRIPT $plain_sdf $enriched_sdf

################################## alternative for big data
# echo "doing spectra generation via CHUNKS - SET manually!!!"
# CHUNK_NAME=$1  # 8M_chunk0 / 2M_derivatized_chunk0 / 12M_derivatized_chunk0
# CHUNK_FOLDER=${INPUT}_sdf_chunks
# plain_sdf=${SPECTRO_DIR}/MassGenie/data/trial_set/tmp/$CHUNK_FOLDER/$CHUNK_NAME
# enriched_sdf=${SPECTRO_DIR}/MassGenie/data/trial_set/tmp/$CHUNK_FOLDER/${CHUNK_NAME}_enriched_smiles.sdf
# after_phase3_pickle=$SPECTRO_DIR/MassGenie/data/trial_set/tmp/$CHUNK_FOLDER/${CHUNK_NAME}_df_after_phase3.pkl
# save_prepared_path=$SPECTRO_DIR/MassGenie/data/trial_set/tmp/$CHUNK_FOLDER/prepared_sets/${CHUNK_NAME}${TOKENIZER}_bart_prepared_data.pkl

# echo ">>> Chunk ${CHUNK_NAME} processing"
# echo Generating SKIPPED
# . $PRED_SCRIPT $plain_sdf $enriched_sdf
# ##################################

conda deactivate
} && {
################### PREPROCESS 3 (FILTERING)

ARGS3="--load-generated-sdf $enriched_sdf
       --save-pickle-path $after_phase3_pickle
"
#--max-peaks

# echo PHASE3 SKIPPED
python data_preprocess3.py $ARGS3

} && {
################### PREPROCESS 4 (TOKENIZATION, DATASET CREATION - ALL THE INPUT LISTS)
ARGS4="--df-path $after_phase3_pickle
       --save-pickle-path $save_prepared_path
       --tokenizer bbpe
"
#${TOKENIZER#?} (tokenizer argument)
# echo PHASE4 SKIPPED
python data_preprocess4.py $ARGS4
}
