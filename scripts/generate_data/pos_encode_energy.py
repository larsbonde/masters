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

model_dir = Path("/home/projects/ht3_aim/people/idamei/data/train_data/")

mock_raw_dir = data_root / "raw" / "energy_terms_mock"
mock_raw_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

new_dir = processed_dir / "energy_terms_pos"
new_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

paths = list(model_dir.glob("*"))

for path in paths:
    raw_idx = int(path.name.split("_")[0])
    mock_raw_path = mock_raw_dir / f"{raw_idx}_tcrpmhc.pdb"
    mock_raw_path.touch(mode=0o664)
    
join_key = [int(x.name.split("_")[0]) for x in paths]
path_df = pd.DataFrame({'#ID': join_key, 'path': paths})
metadata = pd.read_csv(metadata_path)
metadata = metadata.join(path_df.set_index("#ID"), on="#ID", how="inner")  # filter to non-missing data
metadata = metadata.reset_index(drop=True)

mhc_idx = 20
pep_idx = 21
tcra_idx = 22
tcrb_idx = 23
time_idx = [92, 116, 140]
#pos_key to use ["P", "M", "A", "B"]
targets = list()
for i, path, target in zip(list(metadata.index), metadata["path"], metadata["binder"]):
    x = np.load(path)
    
    new_pos = np.array([
        x[:,pep_idx],
        x[:,mhc_idx],
        x[:,tcra_idx],
        x[:,tcrb_idx],
    ]).T
    new_pos = torch.from_numpy(new_pos)
    
    new_x = np.delete(x, [mhc_idx, pep_idx, tcra_idx, tcrb_idx] + time_idx, axis=1)
    new_x = torch.from_numpy(new_x).float()
    new_x = torch.hstack((new_x, new_pos))
    torch.save(new_x, new_dir / f"data_{i}.pt")
    
    targets.append([target])

torch.save(targets, new_dir / f"targets.pt")