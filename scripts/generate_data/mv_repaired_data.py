import os
from pathlib import Path

root_path = Path("/home/projects/ht3_aim/people/idamei/results/energy_calc_full_output")
fold_paths = [root_path / f"{i}" for i in range(5)]

target_root_path = Path("/home/projects/ht3_aim/people/sebdel/masters/data/neat_data/raw/")

for fold_path in fold_paths:
    for pos_neg in ["positives", "negatives"]:
        new_fold_path = fold_path / pos_neg
        model_paths = new_fold_path.glob("*")
        for model_path in model_paths:
            file_idx = model_path.name
            rosetta = model_path / f"{file_idx}_model_0001.pdb"
            foldx = model_path / f"{file_idx}_model_Repair.pdb"

            if rosetta.exists() and file_idx != "12636":
                new_rosetta = target_root_path / "rosetta_repair" / f"{file_idx}_tcrpmhc.pdb"
                os.system(f"cp {rosetta} {new_rosetta}")

            if foldx.exists():
                new_foldx = target_root_path / "foldx_repair" / f"{file_idx}_tcrpmhc.pdb"
                os.system(f"cp {foldx} {new_foldx}")
