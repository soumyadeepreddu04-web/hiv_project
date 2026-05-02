from __future__ import annotations

from typing import Any, Dict, List, Tuple

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors

from dti_pipeline.config import FEATURE_RECIPES, FeatureBlocks, MAX_PARALLEL_WORKERS, RunLogger


ECFP4_BITS = 2048
ATOM_PAIR_BITS = 2048
TORSION_BITS = 2048
ECFP4_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=ECFP4_BITS)
ATOM_PAIR_GENERATOR = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=ATOM_PAIR_BITS)
TORSION_GENERATOR = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=TORSION_BITS)

DESCRIPTOR_FUNCTIONS: Dict[str, Any] = {
    "mol_wt": Descriptors.MolWt,
    "logp": Crippen.MolLogP,
    "mol_mr": Crippen.MolMR,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "hbond_donors": Lipinski.NumHDonors,
    "hbond_acceptors": Lipinski.NumHAcceptors,
    "rotatable_bonds": Lipinski.NumRotatableBonds,
    "heavy_atoms": Lipinski.HeavyAtomCount,
    "hetero_atoms": Descriptors.NumHeteroatoms,
    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
    "ring_count": rdMolDescriptors.CalcNumRings,
    "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings,
    "aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings,
    "valence_electrons": Descriptors.NumValenceElectrons,
}


def featurize_dataset(df: pd.DataFrame, logger: RunLogger) -> FeatureBlocks:
    n_molecules = len(df)
    descriptor_names = list(DESCRIPTOR_FUNCTIONS.keys())
    arrays: Dict[str, np.ndarray] = {
        "ecfp4_2048": np.zeros((n_molecules, ECFP4_BITS), dtype=np.float32),
        "maccs": np.zeros((n_molecules, 167), dtype=np.float32),
        "atom_pair_2048": np.zeros((n_molecules, ATOM_PAIR_BITS), dtype=np.float32),
        "torsion_2048": np.zeros((n_molecules, TORSION_BITS), dtype=np.float32),
        "descriptors": np.zeros((n_molecules, len(descriptor_names)), dtype=np.float32),
    }
    names: Dict[str, List[str]] = {
        "ecfp4_2048": [f"ecfp4_bit_{index}" for index in range(ECFP4_BITS)],
        "maccs": [f"maccs_bit_{index}" for index in range(167)],
        "atom_pair_2048": [f"atom_pair_bit_{index}" for index in range(ATOM_PAIR_BITS)],
        "torsion_2048": [f"torsion_bit_{index}" for index in range(TORSION_BITS)],
        "descriptors": descriptor_names,
    }
    morgan_fingerprints: List[Any] = [None] * n_molecules

    def _process(item: Tuple[int, str]) -> Tuple[int, Any, Any, Any, Any, np.ndarray]:
        row_idx, smiles = item
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES at dataset row {row_idx + 1}: {smiles}")
        ecfp4_fp = ECFP4_GENERATOR.GetFingerprint(mol)
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        atom_pair_fp = ATOM_PAIR_GENERATOR.GetFingerprint(mol)
        torsion_fp = TORSION_GENERATOR.GetFingerprint(mol)
        descriptor_vals = np.asarray([float(func(mol)) for func in DESCRIPTOR_FUNCTIONS.values()], dtype=np.float32)
        return row_idx, ecfp4_fp, maccs_fp, atom_pair_fp, torsion_fp, descriptor_vals

    with logger.section("Generating molecular feature blocks"):
        smiles_enumerated = list(enumerate(df["smiles"].tolist()))
        results = Parallel(n_jobs=MAX_PARALLEL_WORKERS, backend="loky")(
            delayed(_process)(item) for item in smiles_enumerated
        )

        for row_idx, ecfp4_fp, maccs_fp, atom_pair_fp, torsion_fp, descriptor_vals in results:
            DataStructs.ConvertToNumpyArray(ecfp4_fp, arrays["ecfp4_2048"][row_idx])
            DataStructs.ConvertToNumpyArray(maccs_fp, arrays["maccs"][row_idx])
            DataStructs.ConvertToNumpyArray(atom_pair_fp, arrays["atom_pair_2048"][row_idx])
            DataStructs.ConvertToNumpyArray(torsion_fp, arrays["torsion_2048"][row_idx])
            arrays["descriptors"][row_idx] = descriptor_vals
            morgan_fingerprints[row_idx] = ecfp4_fp
            if (row_idx + 1) % 250 == 0:
                logger.log(f"Featurized {row_idx + 1}/{n_molecules} molecules.")

        if n_molecules % 250 != 0:
            logger.log(f"Featurized {n_molecules}/{n_molecules} molecules.")

    return FeatureBlocks(arrays=arrays, names=names, morgan_fingerprints=morgan_fingerprints)


def compose_feature_matrix(blocks: FeatureBlocks, recipe_name: str, row_ids: np.ndarray | None = None) -> Tuple[np.ndarray, List[str]]:
    component_names = FEATURE_RECIPES[recipe_name]
    arrays = [blocks.arrays[name] for name in component_names]
    feature_names = [feature_name for name in component_names for feature_name in blocks.names[name]]
    matrix = np.concatenate(arrays, axis=1)
    if row_ids is not None:
        matrix = matrix[row_ids]
    return matrix, feature_names
