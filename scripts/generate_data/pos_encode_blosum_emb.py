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

blosum_encode_dict = {
    "A":[4, -1, -2, -2 , 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],
    "R":[-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],
    "N":[-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],
    "D":[-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],
    "C":[ 0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],
    "Q":[-1, 1, 0, 0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, 0],
    "E":[-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 0],
    "G":[ 0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],
    "H":[-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],
    "I":[-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],
    "L":[-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],
    "K":[-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],
    "M":[-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],
    "F":[-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],
    "P":[-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],
    "S":[1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2, 0],
    "T":[0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1, 5, -2, -2,  0, 0],
    "W":[-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],
    "Y":[-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2,  2,  7, -1, 0],
    "V":[0, -3, -3, -3, -1, -2, -2, -3, -3,  3, 1, -2, 1, -1, -2, -2,  0, -3, -1, 4, 0],
    "-":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }

root = Path("/home/projects/ht3_aim/people/sebdel/masters/data")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed"
sequence_path = data_root / "full_seqs.fsa"

dataset = LSTMDataset(
    data_dir=processed_dir / "proteinsolver_embeddings_pos", 
    annotations_path=processed_dir / "proteinsolver_embeddings_pos" / "targets.pt"
)
new_esm_dir = processed_dir / "blosum_embeddings_pos"
new_esm_dir.mkdir(mode=0o775, parents=True, exist_ok=True)


with open(sequence_path) as outfile:
    lines = outfile.readlines()
lines = "".join(lines)
lines = lines.split(">")

for i, seq in enumerate(lines[1:]):
    seq = seq.split("\n")[1]
    data = list()
    for res in seq:
        blosum_enc = torch.Tensor(blosum_encode_dict[res])
        data.append(blosum_enc)
    data = torch.stack(data)
    gnn_emb = dataset[i][0]  # get pos enc
    esm_out = torch.hstack((data, gnn_emb[:,-4:]))
    torch.save(esm_out, new_esm_dir / f"data_{i}.pt")
