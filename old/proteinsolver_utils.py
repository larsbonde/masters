from kmtools import structure_tools
from numba import njit
from typing import NamedTuple, Optional

import torch.nn as nn
import torch
import numpy as np


class ProteinData(NamedTuple):
    sequence: str
    row_index: torch.LongTensor
    col_index: torch.LongTensor
    distances: torch.FloatTensor


def extract_seq_and_adj(structure, chain_id, remove_hetatms=False):
    domain, result_df = get_interaction_dataset_wdistances(
        structure, 0, chain_id, r_cutoff=12, remove_hetatms=remove_hetatms
    )
    domain_sequence = structure_tools.get_chain_sequence(domain)
    assert max(result_df["residue_idx_1"].values) < len(domain_sequence)
    assert max(result_df["residue_idx_2"].values) < len(domain_sequence)
    data = ProteinData(
        domain_sequence,
        result_df["residue_idx_1"].values,
        result_df["residue_idx_2"].values,
        result_df["distance"].values,
        # result_df["distance_backbone"].values,
        # result_df["orientation_1"].values,
        # result_df["orientation_2"].values,
        # result_df["orientation_3"].values,
    )
    return data


def get_interaction_dataset_wdistances(
    structure, model_id, chain_id, r_cutoff=12, remove_hetatms=False
):
    chain = structure[0][chain_id]
    num_residues = len(list(chain.residues))
    dd = structure_tools.DomainDef(model_id, chain_id, 1, num_residues)
    domain = structure_tools.extract_domain(structure, [dd], remove_hetatms=remove_hetatms)
    distances_core = structure_tools.get_distances(
        domain.to_dataframe(), r_cutoff, groupby="residue"
    )
    assert (distances_core["residue_idx_1"] <= distances_core["residue_idx_2"]).all()
    return domain, distances_core
	
	
@njit
def seq_to_tensor(seq: bytes) -> np.ndarray:
    amino_acids = [71, 86, 65, 76, 73, 67, 77, 70, 87, 80, 68, 69, 83, 84, 89, 81, 78, 75, 82, 72]
    # skip_char = 46  # ord('.')
    out = np.ones(len(seq)) * 20
    for i, aa in enumerate(seq):
        for j, aa_ref in enumerate(amino_acids):
            if aa == aa_ref:
                out[i] = j
                break
    return out

	
@torch.no_grad()
def get_node_outputs(
    net: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_categories: int = 20,
    output_transform: Optional[str] = None,
    oneshot: bool = False,
) -> torch.Tensor:
    """Return network output for each node in the reference sequence.

    Args:
        net: The network to use for making predictions.
        x: Node attributes for the target sequence.
        edge_index: Edge indices of the target sequence.
        edge_attr: Edge attributes of the target sequence.
        num_categories: The number of categories to which the network assigns individual nodes
            (e.g. the number of amino acids for the protein design problem).
        output_transform: Transformation to apply to network outputs.
            - `None` - No transformation.
            - `proba` - Apply the softmax transformation.
            - `logproba` - Apply the softmax transformation and log the results.
        oneshot: Whether predictions should be made using a single pass through the network,
            or incrementally, by making a single prediction at a time.

    Returns:
        A tensor of network predictions for each node in `x`.
    """
    assert output_transform in [None, "proba", "logproba"]

    x_ref = x
    x = torch.ones_like(x_ref) * num_categories
    x_proba = torch.zeros_like(x_ref).to(torch.float)
    index_array_ref = torch.arange(x_ref.size(0))
    mask = x == num_categories
    while mask.any():
        output = net(x, edge_index, edge_attr)
        if output_transform == "proba":
            output = torch.softmax(output, dim=1)
        elif output_transform == "logproba":
            output = torch.softmax(output, dim=1).log()

        output_for_x = output.gather(1, x_ref.view(-1, 1))

        if oneshot:
            return output_for_x.data.cpu()

        output_for_x = output_for_x[mask]
        index_array = index_array_ref[mask]
        max_proba, max_proba_position = output_for_x.max(dim=0)

        assert x[index_array[max_proba_position]] == num_categories
        assert x_proba[index_array[max_proba_position]] == 0
        correct_amino_acid = x_ref[index_array[max_proba_position]].item()
        x[index_array[max_proba_position]] = correct_amino_acid
        assert output[index_array[max_proba_position], correct_amino_acid] == max_proba
        x_proba[index_array[max_proba_position]] = max_proba
        mask = x == num_categories
    return x_proba.data.cpu()