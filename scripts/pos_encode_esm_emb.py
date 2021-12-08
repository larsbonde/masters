import sys
sys.path.append('/home/projects/ht3_aim/people/sebdel/masters/') # add my repo to python path
import torch
import modules

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
processed_dir = data_root / "processed" / "tcr_binding"
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
out_dir = root / "state_files" / "tcr_binding"

dataset = LSTMDataset(
    data_dir=processed_dir / "gnn_out_pos_128", 
    annotations_path=processed_dir / "gnn_out_pos_128" / "targets.pt"
)
esm_dir = processed_dir / "esm_1b_embeddings/esm_encodings/"
new_esm_dir = processed_dir / "esm_1b_embeddings/esm_encodings_pos/"
new_esm_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

for i in range(len(dataset)):
    esm_file = torch.load(esm_dir / f"{i}.pt")
    esm_emb = list(esm_file["representations"].values())[0]
    gnn_emb = dataset[i][0]  # get pos enc
    
    esm_out = torch.hstack((esm_emb, gnn_emb[:,-4:]))
    
    torch.save(esm_out, new_esm_dir / f"data_{i}.pt")