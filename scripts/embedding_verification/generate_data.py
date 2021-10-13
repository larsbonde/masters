#! /usr/bin/env python3

import sys
import kmbio
import numpy as np
from pathlib import Path
sys.path.append('/home/projects/ht3_aim/people/sebdel/masters/')
import modules
from modules.dataset import *


class ChainFilter(kmbio.PDB.Select):
    def __init__(self, subset):
        self.subset = subset

    def accept_chain(self, chain):
        if chain.id in self.subset:
            return 1
        else:
            return 0


root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "embedding_dataset.csv"

overwrite = False

raw_files, targets = get_data(
    model_dir=data_root / "raw" / "tcrpmhc",
    metadata=data_root / "metadata.csv",
)
mask = np.ma.masked_array(raw_files, mask=targets)  # only get positives

pmhc_chain_subset = ["M", "P"]
p_chain_subset = ["P"]
annotated_paths = list()

outdir_1 = data_root / "raw" / "pmhc"
outdir_2 = data_root / "raw" / "p"

outdir_1.mkdir(parents=True, exist_ok=True)
outdir_2.mkdir(parents=True, exist_ok=True)
for raw_file in raw_files[mask.mask]:

    model_id = raw_file.name.split("_")[0]
    pmhc_file_name = outdir_1 / f"{model_id}_pmhc.pdb"
    p_file_name =  outdir_2/ f"{model_id}_p.pdb"
    
    if overwrite or (not pmhc_file_name.is_file() or not p_file_name.is_file()):
        structure  = kmbio.PDB.load(raw_file)
    
        io = kmbio.PDB.io.PDBIO()
        io.set_structure(structure)
        io.save(pmhc_file_name, ChainFilter(subset=pmhc_chain_subset))
    
        io = kmbio.PDB.io.PDBIO()
        io.set_structure(structure)
        io.save(p_file_name, ChainFilter(subset=p_chain_subset))

    annotated_paths.append([raw_file, "0"])  # add indices of peptide
    annotated_paths.append([pmhc_file_name, "1"])  # add indices of peptide
    annotated_paths.append([p_file_name, "2"])  # add indices of peptide

with open(metadata_path, "w") as metadata_outfile:
    for data in annotated_paths:
        print(data[0], data[1], sep=",", file=metadata_outfile)
