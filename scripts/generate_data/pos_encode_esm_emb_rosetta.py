import sys
sys.path.append('/home/projects/ht3_aim/people/sebdel/masters/') # add my repo to python path
import torch
import modules
import pandas as pd

from pathlib import Path

from modules.dataset_utils import *
from modules.dataset import *
from modules.utils import *
from modules.models import *
from modules.lstm_utils import *

np.random.seed(0)
torch.manual_seed(0)

root = Path("/home/projects/ht3_aim/people/sebdel/masters/data")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed"
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
model_dir = data_root / "raw" / "rosetta_repair"

dataset = LSTMDataset(
    data_dir=processed_dir / "proteinsolver_embeddings_pos_rosetta_repair", 
    annotations_path=processed_dir / "proteinsolver_embeddings_pos_rosetta_repair" / "targets.pt"
)
esm_dir = processed_dir / "esm_embeddings"

# Get raw to processed data idx map for repaired structures
paths_repair = list(model_dir.glob("*"))
join_key_repair = [int(x.name.split("_")[0]) for x in paths_repair]
path_df_repair = pd.DataFrame({'#ID': join_key_repair, 'path_repair': paths_repair})
path_df_repair = path_df_repair.sort_values(by="#ID", ignore_index=True)


path_df_repair["repair_idx"] = path_df_repair.index

# Get raw to processed data idx for esm embeddings
esm_source_model_dir = data_root / "raw" / "tcrpmhc"
paths_esm = list(esm_source_model_dir.glob("*"))
join_key_esm = [int(x.name.split("_")[0]) for x in paths_esm]
path_df_esm = pd.DataFrame({'#ID': join_key_esm, 'path_tcrpmhc': paths_esm})
path_df_esm = path_df_esm.sort_values(by="#ID", ignore_index=True)
path_df_esm["esm_idx"] = path_df_esm.index

mapped_df = path_df_repair.join(path_df_esm.set_index("#ID"), on="#ID", how="inner")

# go row for row, get esm and repair idx and get appropriate files and merge

esm_ps_dir = processed_dir / "proteinsolver_esm_embeddings_pos_rosetta_repair"
esm_ps_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

for esm_idx, rep_idx in zip(mapped_df["esm_idx"], mapped_df["repair_idx"]):
    esm_file = torch.load(esm_dir / f"{esm_idx}.pt")
    esm_emb = list(esm_file["representations"].values())[0] 
    gnn_emb = dataset[rep_idx][0]
    esm_ps_out = torch.hstack((esm_emb, gnn_emb))
    torch.save(esm_ps_out, esm_ps_dir / f"data_{rep_idx}.pt")  

