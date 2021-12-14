#! /usr/bin/env python3

import os

from_dir = "/home/projects/ht3_aim/data/210916_TCRpMHCmodels/models/"
to_dir = "/home/projects/ht3_aim/people/sebdel/masters/data/neat_data/raw/tcrpmhc/"
model_suffix = "model_TCR-pMHC.pdb"
for subdir in os.listdir(from_dir):
    subdir_id = subdir.split("_")[0]
    new_name = f"{subdir_id}_tcrpmhc.pdb"
    os.system(f"cp {from_dir}/{subdir}/{model_suffix} {to_dir}/{new_name}")
