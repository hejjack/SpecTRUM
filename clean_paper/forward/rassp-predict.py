#!/usr/bin/env python3

use_gpu=True
batch_size=8

import argparse
import json
import rassp
from rassp import netutil
from rdkit import Chem
import numpy as np

from rassp import msutil,model
import sys
sys.modules['msutil'] = msutil
sys.modules['model'] = model


p = argparse.ArgumentParser()
p.add_argument('-m','--model')
p.add_argument('-e','--meta')
p.add_argument('-s','--smiles')
p.add_argument('-o','--output')
p.add_argument('-w','--workers', default=4)

a = p.parse_args()

predictor = netutil.PredModel(
    a.meta,
    a.model,
    USE_CUDA=use_gpu,
    data_parallel=False,
)

with open(a.smiles) as sf:
	smiles = sf.read().splitlines()

mols = [ Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles]
assert len(mols) == len(smiles)

pred = predictor.pred(
    mols,
    progress_bar=False,
    normalize_pred=True,
    output_hist_bins=True,
    batch_size=batch_size, # XXX
    dataloader_config={
        'pin_memory': False,
        'num_workers': int(a.workers), # XXX
        'persistent_workers': False,
    },
    benchmark_dataloader=False,
)['pred_binned']

assert len(pred) == len(smiles)

with open(a.output,'w') as out:
    for i,s in enumerate(pred):
        out.write(json.dumps({ 'mz' : list(s[:,0].astype(float)), 'intensity' : list(s[:,1].astype(float)), 'smiles' : smiles[i]}))
        out.write('\n')

