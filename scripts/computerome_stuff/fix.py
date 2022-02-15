import torch
from pathlib import Path

d_dir = Path("/home/people/sebdel/ht3_aim/masters/data/neat_data/processed/energy_terms_pos/")
paths = list(d_dir.glob("*"))
for path in paths:
    t = torch.load(path)
    if type(t) != list:
        torch.save(t.float(), path)

