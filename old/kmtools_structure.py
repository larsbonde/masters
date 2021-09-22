import pandas as pd
import numpy as np

from typing import NamedTuple, List, Optional
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
#from Bio.PDB.Structure import Structure
from Bio.PDB.Polypeptide import Polypeptide
from scipy.spatial import cKDTree
from kmtools_constants import *

# Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""The structure class, representing a macromolecular structure."""
from typing import List, NamedTuple

import numpy as np
import pandas as pd

from .entity import Entity


class StructureRow(NamedTuple):
    structure_id: str
    model_idx: int
    model_id: int
    chain_idx: int
    chain_id: str
    residue_idx: int
    residue_id_0: str
    residue_id_1: int
    residue_id_2: str
    residue_resname: str
    residue_segid: int
    atom_idx: int
    atom_name: str
    atom_fullname: str
    atom_x: float
    atom_y: float
    atom_z: float
    atom_bfactor: float
    atom_occupancy: float
    atom_altloc: str
    atom_serial_number: int
    #: Additional covalent bonds (like disulphide bonds).
    atom_extra_bonds: List[int]


class Structure(Entity):
    """
    The Structure class contains a collection of Model instances.
    """

    level = "S"

    def __repr__(self):
        return "<Structure id=%s>" % self.id

    def __lt__(self, other):
        return self.id.lower() < other.id.lower()

    def __le__(self, other):
        return self.id.lower() <= other.id.lower()

    def __eq__(self, other):
        return self.id.lower() == other.id.lower()

    def __ne__(self, other):
        return self.id.lower() != other.id.lower()

    def __ge__(self, other):
        return self.id.lower() >= other.id.lower()

    def __gt__(self, other):
        return self.id.lower() > other.id.lower()

    def extract_models(self, model_ids):
        # TODO: Not sure if this is neccessary
        structure = Structure(self.id)
        for model_id in model_ids:
            structure.add(self[model_id].copy())
        return structure

    def select(self, models=None, chains=None, residues=None, hetatms=None):
        """This method allows you to select things from structures using a variety of queries.

        In particular, you should be able to select one or more chains,
        and all HETATMs that are within a certain distance of those chains.
        """
        raise NotImplementedError

    def to_dataframe(self) -> pd.DataFrame:
        """Convert this structure into a pandas DataFrame."""
        data = []
        model_idx, chain_idx, residue_idx, atom_idx = -1, -1, -1, -1
        for model in self.models:
            model_idx += 1
            for chain in model.chains:
                chain_idx += 1
                for residue in chain.residues:
                    residue_idx += 1
                    for atom in residue.atoms:
                        atom_idx += 1
                        data.append(
                            StructureRow(
                                structure_id=self.id,
                                model_idx=model_idx,
                                model_id=model.id,
                                chain_idx=chain_idx,
                                chain_id=chain.id,
                                residue_idx=residue_idx,
                                residue_id_0=residue.id[0],
                                residue_id_1=residue.id[1],
                                residue_id_2=residue.id[2],
                                residue_resname=residue.resname,
                                residue_segid=residue.segid,
                                atom_idx=atom_idx,
                                atom_name=atom.name,
                                atom_fullname=atom.fullname,
                                atom_x=atom.coord[0],
                                atom_y=atom.coord[1],
                                atom_z=atom.coord[2],
                                atom_bfactor=atom.bfactor,
                                atom_occupancy=atom.occupancy,
                                atom_altloc=atom.altloc,
                                atom_serial_number=atom.serial_number,
                                atom_extra_bonds=[],
                            )
                        )
        df = pd.DataFrame(data)
        return df

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "Structure":
        """Generate a new structure from a dataframe of atoms and an array of bonds.

        Warning:
            - If the `df` DataFrame was loaded from a CSV file using :any:`pandas.read_csv`,
              you *must* specify ``na_values=[""]`` and ``keep_default_na=False`` in order to get
              correct results. Otherwise, ``NA`` atoms may be intepreted as nulls.

        Args:
            df: DataFrame which should be converted to a Structure.

        Returns:
            structure: A :any:`Structure` object containing information present in the DataFrame.
        """
        from kmbio.PDB.core.model import Model
        from kmbio.PDB.core.chain import Chain
        from kmbio.PDB.core.residue import Residue
        from kmbio.PDB.core.atom import Atom

        assert (df["structure_id"] == df["structure_id"].iloc[0]).all()
        structure = Structure(df["structure_id"].iloc[0])
        # Groupby skips rows with NAs
        structure_df = df.drop(columns=["structure_id"])
        for (_, model_id), model_df in _groupby(structure_df, ["model_idx", "model_id"]):
            model = Model(model_id)
            structure.add(model)
            for (_, chain_id), chain_df in _groupby(model_df, ["chain_idx", "chain_id"]):
                chain = Chain(chain_id)
                model.add(chain)
                for (
                    (_, residue_id_0, residue_id_1, residue_id_2, residue_resname, residue_segid),
                    residue_df,
                ) in _groupby(
                    chain_df,
                    [
                        "residue_idx",
                        "residue_id_0",
                        "residue_id_1",
                        "residue_id_2",
                        "residue_resname",
                        "residue_segid",
                    ],
                ):
                    residue = Residue(
                        (residue_id_0, residue_id_1, residue_id_2),
                        resname=residue_resname,
                        segid=residue_segid,
                    )
                    chain.add(residue)
                    for (_, atom_name), atom_df in _groupby(residue_df, ["atom_idx", "atom_name"]):
                        assert len(atom_df) == 1
                        atom_s = atom_df.iloc[0]
                        atom = Atom(
                            name=atom_name,
                            coord=(atom_s.atom_x, atom_s.atom_y, atom_s.atom_z),
                            bfactor=atom_s.atom_bfactor,
                            occupancy=atom_s.atom_occupancy,
                            altloc=atom_s.atom_altloc,
                            fullname=atom_s.atom_fullname,
                            serial_number=atom_s.atom_serial_number,
                        )
                        residue.add(atom)
        return structure

    @property
    def models(self):
        for m in self:
            yield m

    @property
    def chains(self):
        for m in self:
            for c in m:
                yield c

    @property
    def residues(self):
        for c in self.chains:
            for r in c:
                yield r

    @property
    def atoms(self):
        for r in self.residues:
            for a in r:
                yield a


def _groupby(df, columns, *args, **kwargs):
    """Groupby columns, *not* ignoring rows containing NANs.

    Have to use this until pandas-dev/pandas#3729 is fixed.
    """
    if df[columns].isnull().any().any():
        assert not (df[columns] == np.inf).any().any()
        df[columns] = df[columns].fillna(np.inf)
        for group_key, group_df in df.groupby(columns, *args, **kwargs):
            group_key = [(k if k != np.inf else np.nan) for k in group_key]
            yield group_key, group_df
    else:
        for group_key, group_df in df.groupby(columns, *args, **kwargs):
            yield group_key, group_df


# kmtools.structure_tools
class DomainDef(NamedTuple):
    model_id: int
    chain_id: str
    #: 1-based
    domain_start: int
    domain_end: int

	
def get_chain_sequence(
    chain: Chain, if_unknown: str = "error", unknown_residue_marker: str = "X"
) -> str:
    # UNKNOWN_MODE = Literal["error", "replace"]
    chain_aa_list = []
    for residue in chain.residues:
        aaa = RESIDUE_MAPPING_TO_CANONICAL.get(residue.resname)
        if aaa is not None and aaa in AAA_DICT:
            aa = AAA_DICT[aaa]
        elif if_unknown == "replace":
            aa = unknown_residue_marker
        else:
            raise ValueError(
                f"Cound not convert residue '{residue.resname}' to a single amino acid code. "
                f"The canonical resname of the residue is '{aaa}'."
            )
        chain_aa_list.append(aa)
    assert len(chain_aa_list) == len(list(chain.residues))
    return "".join(chain_aa_list)

	
def extract_domain(
    structure: Structure, dds: List[DomainDef], remove_hetatms=False, hetatm_residue_cutoff=None
) -> Structure:
    if hetatm_residue_cutoff is not None:
        # Keep residues from HETATM chain that are within `hetatm_residue_cutoff`
        # from any atom in the extracted chain.
        raise NotImplementedError
    assert len({(dd.model_id, dd.chain_id) for dd in dds}) == len(dds)
    new_structure = Structure(structure.id)
    new_model = Model(0)
    new_structure.add(new_model)
    for dd in dds:
        new_chain = Chain(dd.chain_id)
        new_model.add(new_chain)
        residues = list(structure[dd.model_id][dd.chain_id].get_residues())
        domain_residues = residues[dd.domain_start - 1 : dd.domain_end]
        if remove_hetatms:
            domain_residues = [r for r in domain_residues if not r.id[0].strip()]
        (new_chain.add(residue) for residue in domain_residues)
    return new_structure


def get_distances(
    structure_df: pd.DataFrame, max_cutoff: Optional[float], groupby: str = "atom"
) -> pd.DataFrame:
    """Process structure dataframe to extract interacting chains, residues, or atoms.

    Args:
        structure_df: Structure dataframe, as returned by `kmbio.PDB.Structure.to_dataframe`.
        max_cutoff: Maximum distance for inclusion in results (Angstroms).
        groupby: Which pairs of objects to return?
            (Possible options are: `chain{,-backbone,-ca}`, `residue{,-backbone,-ca}`, `atom`).
    """
    assert groupby in [
        "chain",
        "chain-backbone",
        "chain-ca",
        "chain-cb",
        "residue",
        "residue-backbone",
        "residue-ca",
        "residue-cb",
        "atom",
    ]

    residue_idxs = set(structure_df["residue_idx"])
    if groupby.endswith("-backbone"):
        structure_df = structure_df[
            (structure_df["atom_name"] == "N")
            | (structure_df["atom_name"] == "CA")
            | (structure_df["atom_name"] == "C")
        ]
    elif groupby.endswith("-ca"):
        structure_df = structure_df[(structure_df["atom_name"] == "CA")]
    elif groupby.endswith("-cb"):
        structure_df = structure_df[(structure_df["atom_name"] == "CB")]
    assert not residue_idxs - set(structure_df["residue_idx"])

    pairs_df = get_atom_distances(structure_df, max_cutoff=max_cutoff)
    annotate_atom_distances(pairs_df, structure_df)

    if groupby.startswith("residue"):
        pairs_df = _groupby_residue(pairs_df)
    elif groupby.startswith("chain"):
        pairs_df = _groupby_chain(pairs_df)
    return pairs_df


def get_atom_distances(structure_df: pd.DataFrame, max_cutoff: Optional[float]) -> pd.DataFrame:
    if max_cutoff is None:
        return get_atom_distances_dense(structure_df)
    else:
        return get_atom_distances_sparse(structure_df, max_cutoff)


def get_atom_distances_dense(structure_df: pd.DataFrame) -> pd.DataFrame:
    num_atoms = len(structure_df)
    xyz = structure_df[["atom_x", "atom_y", "atom_z"]].values
    xyz_by_xyz = xyz[:, None, :] - xyz[None, :, :]
    xyz_by_xzy_dist = np.sqrt((xyz_by_xyz ** 2).sum(axis=2))
    assert xyz_by_xzy_dist.ndim == 2
    row = np.repeat(np.arange(num_atoms), num_atoms)
    col = np.tile(np.arange(num_atoms), num_atoms)
    atom_idx = structure_df["atom_idx"].values
    pairs_df = pd.DataFrame(
        {
            "atom_idx_1": atom_idx[row],
            "atom_idx_2": atom_idx[col],
            "distance": xyz_by_xzy_dist.reshape(-1),
        }
    )
    pairs_df = pairs_df[pairs_df["atom_idx_1"] < pairs_df["atom_idx_2"]]
    return pairs_df


def get_atom_distances_sparse(structure_df: pd.DataFrame, max_cutoff: float) -> pd.DataFrame:
    xyz = structure_df[["atom_x", "atom_y", "atom_z"]].values
    tree = cKDTree(xyz)
    coo_mat = tree.sparse_distance_matrix(tree, max_distance=max_cutoff, output_type="coo_matrix")
    assert coo_mat.row.max() < xyz.shape[0] and coo_mat.col.max() < xyz.shape[0]
    atom_idx = structure_df["atom_idx"].values
    pairs_df = pd.DataFrame(
        {
            "atom_idx_1": atom_idx[coo_mat.row],
            "atom_idx_2": atom_idx[coo_mat.col],
            "distance": coo_mat.data,
        }
    )
    pairs_df = pairs_df[pairs_df["atom_idx_1"] < pairs_df["atom_idx_2"]]
    return pairs_df


def annotate_atom_distances(pairs_df: pd.DataFrame, structure_df: pd.DataFrame) -> None:
    atom_to_residue_map = {
        row.atom_idx: (row.model_idx, row.chain_idx, row.residue_idx)
        for row in structure_df.itertuples()
    }
    for suffix in ["_1", "_2"]:
        (
            pairs_df[f"model_idx{suffix}"],
            pairs_df[f"chain_idx{suffix}"],
            pairs_df[f"residue_idx{suffix}"],
        ) = list(zip(*pairs_df[f"atom_idx{suffix}"].map(atom_to_residue_map)))


def _groupby_residue(pairs_df: pd.DataFrame) -> pd.DataFrame:
    residue_pairs_df = (
        pairs_df.groupby(["residue_idx_1", "residue_idx_2"]).agg({"distance": min}).reset_index()
    )
    residue_pairs_df = residue_pairs_df[
        (residue_pairs_df["residue_idx_1"] != residue_pairs_df["residue_idx_2"])
    ]
    return residue_pairs_df


def _groupby_chain(pairs_df: pd.DataFrame) -> pd.DataFrame:
    chain_pairs_df = (
        pairs_df.groupby(["chain_idx_1", "chain_idx_2"]).agg({"distance": min}).reset_index()
    )
    chain_pairs_df = chain_pairs_df[
        (chain_pairs_df["chain_idx_1"] != chain_pairs_df["chain_idx_2"])
    ]
    return chain_pairs_df


def complete_distances(distance_df: pd.DataFrame) -> pd.DataFrame:
    """Complete `distances_df` so that it corresponds to a symetric dataframe with a zero diagonal.

    Args:
        distance_df: A dataframe produced by `get_distances`, where
            'residue_idx_1' > 'residue_idx_2'.

    Returns:
        A dataframe which corresponds to a dense, symmetric adjacency matrix.
    """
    residues = sorted(
        {r for r in distance_df["residue_idx_1"]} | {r for r in distance_df["residue_idx_2"]}
    )
    complete_distance_df = pd.concat(
        [
            distance_df,
            distance_df.rename(
                columns={"residue_idx_1": "residue_idx_2", "residue_idx_2": "residue_idx_1"}
            ),
            pd.DataFrame({"residue_idx_1": residues, "residue_idx_2": residues, "distance": 0}),
        ],
        sort=False,
    ).sort_values(["residue_idx_1", "residue_idx_2"])
    assert len(complete_distance_df) == len(distance_df) * 2 + len(residues)
    return complete_distance_df
