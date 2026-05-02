from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from dti_pipeline.config import (
    ACTIVE_PCHEMBL_THRESHOLD,
    CHEMBL_BASE_URL,
    CHEMBL_TARGET_IDS,
    CLASSIFICATION_EXTERNAL_PATH,
    CLASSIFICATION_INTERNAL_TRAIN_PATH,
    CLASSIFICATION_SPLIT_MANIFEST_PATH,
    CLASSIFICATION_TEST_PATH,
    CLASSIFICATION_TRAIN_PATH,
    DATASET_CACHE_PATH,
    DOCUMENT_BATCH_SIZE,
    DOCUMENT_CACHE_PATH,
    INACTIVE_PCHEMBL_THRESHOLD,
    MAX_PCHEMBL_SPREAD,
    MIN_CACHED_SMILES_FOR_OFFLINE_BUILD,
    MOLECULE_BATCH_SIZE,
    PAGE_SIZE,
    RAW_CACHE_PATH,
    RANDOM_SEED,
    REGRESSION_TEST_PATH,
    REGRESSION_TRAIN_PATH,
    REQUEST_MAX_RETRIES,
    REQUEST_SLEEP_SECONDS,
    RunLogger,
    SMILES_CACHE_PATH,
    STANDARD_TYPES,
    STANDARD_UNITS,
    TEMPORAL_CLASSIFICATION_TEST_PATH,
    TEMPORAL_CLASSIFICATION_TRAIN_PATH,
    TEMPORAL_REGRESSION_TEST_PATH,
    TEMPORAL_REGRESSION_TRAIN_PATH,
    TEST_SIZE,
)


UNCHARGER = rdMolStandardize.Uncharger()


def fetch_json(url: str, logger: RunLogger | None = None) -> Dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(REQUEST_MAX_RETRIES):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; hiv-protease-dti/4.0)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if logger is not None:
                logger.log(f"Request retry {attempt + 1}/{REQUEST_MAX_RETRIES} failed for {url}: {error}")
            if attempt == REQUEST_MAX_RETRIES - 1:
                break
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch ChEMBL URL after retries: {url}") from last_error


def build_activity_query_url(target_id: str, standard_type: str) -> str:
    params = {
        "target_chembl_id": target_id,
        "standard_type": standard_type,
        "standard_units": STANDARD_UNITS,
        "limit": PAGE_SIZE,
    }
    return f"{CHEMBL_BASE_URL}/activity.json?{urllib.parse.urlencode(params)}"


def paginate_chembl_records(start_url: str, list_key: str, logger: RunLogger | None = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    url = start_url
    page_number = 0

    while url:
        payload = fetch_json(url, logger=logger)
        page_records = payload[list_key]
        records.extend(page_records)
        page_number += 1
        if logger is not None:
            logger.log(f"Fetched page {page_number} with {len(page_records)} {list_key}; running total = {len(records)}")
        next_page = payload["page_meta"]["next"]
        url = f"https://www.ebi.ac.uk{next_page}" if next_page else None
        time.sleep(REQUEST_SLEEP_SECONDS)

    return records


def download_raw_activities(logger: RunLogger) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for target_id in CHEMBL_TARGET_IDS:
        for standard_type in STANDARD_TYPES:
            logger.log(f"Downloading raw activities for target={target_id}, standard_type={standard_type}")
            url = build_activity_query_url(target_id, standard_type)
            records.extend(paginate_chembl_records(url, list_key="activities", logger=logger))

    raw_df = pd.DataFrame(records)
    desired_columns = [
        "activity_id",
        "assay_chembl_id",
        "molecule_chembl_id",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "target_chembl_id",
        "document_chembl_id",
        "activity_comment",
        "data_validity_comment",
        "data_validity_description",
        "potential_duplicate",
    ]
    keep_columns = [column for column in desired_columns if column in raw_df.columns]
    return raw_df[keep_columns].copy()


def load_or_download_raw_activities(logger: RunLogger) -> Tuple[pd.DataFrame, bool]:
    if RAW_CACHE_PATH.exists():
        logger.log(f"Loading cached raw activity table from {RAW_CACHE_PATH.name}")
        return pd.read_csv(RAW_CACHE_PATH), False

    with logger.section("Downloading raw ChEMBL activity table"):
        raw_df = download_raw_activities(logger)
    raw_df.to_csv(RAW_CACHE_PATH, index=False)
    return raw_df, True


def chunked(items: Sequence[str], chunk_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def load_smiles_cache() -> Dict[str, str]:
    smiles_map: Dict[str, str] = {}
    candidate_paths = [
        SMILES_CACHE_PATH,
        DATASET_CACHE_PATH,
        CLASSIFICATION_TRAIN_PATH,
        CLASSIFICATION_TEST_PATH,
    ]

    for path in candidate_paths:
        if not path.exists():
            continue
        cache_df = pd.read_csv(path)
        if {"molecule_chembl_id", "smiles"}.issubset(cache_df.columns):
            partial_map = dict(zip(cache_df["molecule_chembl_id"].astype(str), cache_df["smiles"].astype(str)))
            smiles_map.update(partial_map)

    return smiles_map


def save_smiles_cache(smiles_map: Dict[str, str]) -> None:
    cache_df = pd.DataFrame(
        {
            "molecule_chembl_id": list(smiles_map.keys()),
            "smiles": list(smiles_map.values()),
        }
    ).sort_values("molecule_chembl_id")
    cache_df.to_csv(SMILES_CACHE_PATH, index=False)


def fetch_smiles_batch(batch: Sequence[str], logger: RunLogger | None = None) -> Dict[str, str]:
    params = {"molecule_chembl_id__in": ",".join(batch), "limit": len(batch)}
    url = f"{CHEMBL_BASE_URL}/molecule.json?{urllib.parse.urlencode(params)}"

    try:
        payload = fetch_json(url, logger=logger)
    except RuntimeError:
        if len(batch) == 1:
            return {}
        midpoint = len(batch) // 2
        left_map = fetch_smiles_batch(batch[:midpoint], logger=logger)
        right_map = fetch_smiles_batch(batch[midpoint:], logger=logger)
        return {**left_map, **right_map}

    batch_map: Dict[str, str] = {}
    for molecule in payload["molecules"]:
        molecule_id = molecule.get("molecule_chembl_id")
        structures = molecule.get("molecule_structures") or {}
        smiles = structures.get("canonical_smiles")
        if molecule_id and smiles:
            batch_map[str(molecule_id)] = str(smiles)
    return batch_map


def fetch_smiles_map(molecule_ids: Sequence[str], logger: RunLogger) -> Dict[str, str]:
    smiles_map = load_smiles_cache()
    missing_ids = [molecule_id for molecule_id in molecule_ids if molecule_id not in smiles_map]
    logger.log(
        f"SMILES cache contains {len(smiles_map)} entries; need {len(missing_ids)} additional molecules for this build."
    )

    if not missing_ids:
        return smiles_map

    if len(smiles_map) >= MIN_CACHED_SMILES_FOR_OFFLINE_BUILD:
        logger.log("Large local SMILES cache detected. Proceeding with cached structures and dropping any unresolved tail.")
        return smiles_map

    for batch_number, batch in enumerate(chunked(list(missing_ids), MOLECULE_BATCH_SIZE), start=1):
        smiles_map.update(fetch_smiles_batch(batch, logger=logger))
        save_smiles_cache(smiles_map)
        logger.log(f"Fetched SMILES batch {batch_number}; cache now holds {len(smiles_map)} molecules.")
        time.sleep(REQUEST_SLEEP_SECONDS)

    return smiles_map


def load_document_cache() -> Dict[str, Dict[str, Any]]:
    if not DOCUMENT_CACHE_PATH.exists():
        return {}

    cached = pd.read_csv(DOCUMENT_CACHE_PATH)
    if "document_chembl_id" not in cached.columns:
        return {}

    records: Dict[str, Dict[str, Any]] = {}
    for _, row in cached.iterrows():
        records[str(row["document_chembl_id"])] = {
            "year": row.get("year", np.nan),
            "journal": row.get("journal", ""),
            "title": row.get("title", ""),
        }
    return records


def save_document_cache(document_map: Dict[str, Dict[str, Any]]) -> None:
    rows = []
    for document_id, metadata in document_map.items():
        rows.append(
            {
                "document_chembl_id": document_id,
                "year": metadata.get("year", np.nan),
                "journal": metadata.get("journal", ""),
                "title": metadata.get("title", ""),
            }
        )
    pd.DataFrame(rows).sort_values("document_chembl_id").to_csv(DOCUMENT_CACHE_PATH, index=False)


def fetch_document_batch(batch: Sequence[str], logger: RunLogger | None = None) -> Dict[str, Dict[str, Any]]:
    params = {"document_chembl_id__in": ",".join(batch), "limit": len(batch)}
    url = f"{CHEMBL_BASE_URL}/document.json?{urllib.parse.urlencode(params)}"

    try:
        payload = fetch_json(url, logger=logger)
    except RuntimeError:
        if len(batch) == 1:
            return {}
        midpoint = len(batch) // 2
        left = fetch_document_batch(batch[:midpoint], logger=logger)
        right = fetch_document_batch(batch[midpoint:], logger=logger)
        return {**left, **right}

    document_map: Dict[str, Dict[str, Any]] = {}
    for document in payload.get("documents", []):
        document_id = document.get("document_chembl_id")
        if not document_id:
            continue
        document_map[str(document_id)] = {
            "year": document.get("year", np.nan),
            "journal": document.get("journal", ""),
            "title": document.get("title", ""),
        }
    return document_map


def fetch_document_metadata(document_ids: Sequence[str], logger: RunLogger, allow_network: bool) -> Dict[str, Dict[str, Any]]:
    document_map = load_document_cache()
    missing_ids = [document_id for document_id in document_ids if document_id not in document_map]
    logger.log(
        f"Document cache contains {len(document_map)} entries; need metadata for {len(missing_ids)} additional documents."
    )

    if not missing_ids:
        return document_map

    if not allow_network:
        logger.log("Skipping document year fetch because this run is operating from local cache only.")
        return document_map

    for batch_number, batch in enumerate(chunked(list(missing_ids), DOCUMENT_BATCH_SIZE), start=1):
        document_map.update(fetch_document_batch(batch, logger=logger))
        save_document_cache(document_map)
        logger.log(f"Fetched document batch {batch_number}; cache now holds {len(document_map)} document records.")
        time.sleep(REQUEST_SLEEP_SECONDS)

    return document_map


def standardize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    mol = UNCHARGER.uncharge(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def safe_standardize_smiles(smiles: str) -> str | None:
    try:
        return standardize_smiles(smiles)
    except ValueError:
        return None


def label_from_pchembl(median_pchembl: float) -> float:
    if median_pchembl >= ACTIVE_PCHEMBL_THRESHOLD:
        return 1.0
    if median_pchembl <= INACTIVE_PCHEMBL_THRESHOLD:
        return 0.0
    return np.nan


def scaffold_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold if scaffold else smiles


def clean_activity_table(raw_df: pd.DataFrame, logger: RunLogger) -> pd.DataFrame:
    cleaned = raw_df.copy()
    logger.log(f"Cleaning raw activity table with {len(cleaned)} rows.")
    cleaned["standard_value_nM"] = pd.to_numeric(cleaned["standard_value"], errors="coerce")
    cleaned["pchembl_value"] = pd.to_numeric(cleaned["pchembl_value"], errors="coerce")
    cleaned["potential_duplicate"] = pd.to_numeric(cleaned.get("potential_duplicate"), errors="coerce")

    cleaned = cleaned.dropna(subset=["molecule_chembl_id", "standard_value_nM"])
    logger.log(f"After dropping missing molecule/value rows: {len(cleaned)} rows")

    cleaned = cleaned[cleaned["standard_units"] == STANDARD_UNITS]
    cleaned = cleaned[np.isfinite(cleaned["standard_value_nM"])]
    cleaned = cleaned[cleaned["standard_value_nM"] > 0]
    logger.log(f"After standard-unit and positive-value filters: {len(cleaned)} rows")

    cleaned["standard_relation"] = cleaned["standard_relation"].fillna("=")
    cleaned = cleaned[cleaned["standard_relation"] == "="]
    logger.log(f"After exact-measurement filter: {len(cleaned)} rows")

    if "data_validity_comment" in cleaned.columns:
        cleaned = cleaned[cleaned["data_validity_comment"].isna()]
        logger.log(f"After invalid-data filter: {len(cleaned)} rows")

    if "potential_duplicate" in cleaned.columns:
        cleaned = cleaned[(cleaned["potential_duplicate"].isna()) | (cleaned["potential_duplicate"] == 0)]
        logger.log(f"After duplicate filter: {len(cleaned)} rows")

    cleaned["resolved_pchembl"] = cleaned["pchembl_value"]
    missing_mask = ~np.isfinite(cleaned["resolved_pchembl"])
    cleaned.loc[missing_mask, "resolved_pchembl"] = 9.0 - np.log10(cleaned.loc[missing_mask, "standard_value_nM"])
    cleaned = cleaned[np.isfinite(cleaned["resolved_pchembl"])]
    logger.log(f"After pChEMBL resolution: {len(cleaned)} rows")
    return cleaned.reset_index(drop=True)


def build_curated_dataset(raw_df: pd.DataFrame, logger: RunLogger, allow_document_fetch: bool) -> pd.DataFrame:
    cleaned = clean_activity_table(raw_df, logger)
    cleaned["document_year"] = np.nan

    if "document_chembl_id" in cleaned.columns:
        document_ids = sorted(cleaned["document_chembl_id"].dropna().astype(str).unique())
        document_map = fetch_document_metadata(document_ids, logger, allow_network=allow_document_fetch)
        if document_map:
            cleaned["document_year"] = cleaned["document_chembl_id"].astype(str).map(
                lambda document_id: document_map.get(document_id, {}).get("year", np.nan)
            )
            cleaned["document_year"] = pd.to_numeric(cleaned["document_year"], errors="coerce")
            logger.log(
                f"Attached document years for {cleaned['document_year'].notna().sum()} activity rows across "
                f"{cleaned['document_year'].dropna().nunique()} publication years."
            )

    per_molecule = (
        cleaned.groupby("molecule_chembl_id")
        .agg(
            median_activity_nM=("standard_value_nM", "median"),
            min_activity_nM=("standard_value_nM", "min"),
            max_activity_nM=("standard_value_nM", "max"),
            median_pchembl=("resolved_pchembl", "median"),
            min_pchembl=("resolved_pchembl", "min"),
            max_pchembl=("resolved_pchembl", "max"),
            pchembl_std=("resolved_pchembl", "std"),
            measurement_count=("activity_id", "size"),
            assay_count=("assay_chembl_id", "nunique"),
            document_count=("document_chembl_id", "nunique"),
            first_document_year=("document_year", "min"),
            latest_document_year=("document_year", "max"),
        )
        .reset_index(drop=False)
    )

    per_molecule["pchembl_std"] = per_molecule["pchembl_std"].fillna(0.0)
    per_molecule["label"] = per_molecule["median_pchembl"].apply(label_from_pchembl)
    has_label_conflict = (per_molecule["max_pchembl"] >= ACTIVE_PCHEMBL_THRESHOLD) & (
        per_molecule["min_pchembl"] <= INACTIVE_PCHEMBL_THRESHOLD
    )
    per_molecule = per_molecule[~has_label_conflict].copy()
    per_molecule = per_molecule[per_molecule["pchembl_std"] <= MAX_PCHEMBL_SPREAD].copy()
    logger.log(f"After molecule-level conflict and spread filtering: {len(per_molecule)} unique molecules remain")

    molecule_ids = sorted(per_molecule["molecule_chembl_id"].astype(str).unique())
    smiles_map = fetch_smiles_map(molecule_ids, logger)
    per_molecule["smiles"] = per_molecule["molecule_chembl_id"].astype(str).map(smiles_map)
    per_molecule = per_molecule.dropna(subset=["smiles"]).copy()
    per_molecule["canonical_smiles"] = per_molecule["smiles"].apply(safe_standardize_smiles)
    per_molecule = per_molecule.dropna(subset=["canonical_smiles"]).copy()
    logger.log(f"After SMILES resolution and standardization: {len(per_molecule)} molecules remain")

    grouped = (
        per_molecule.groupby("canonical_smiles")
        .agg(
            median_activity_nM=("median_activity_nM", "median"),
            min_activity_nM=("min_activity_nM", "min"),
            max_activity_nM=("max_activity_nM", "max"),
            median_pchembl=("median_pchembl", "median"),
            min_pchembl=("min_pchembl", "min"),
            max_pchembl=("max_pchembl", "max"),
            pchembl_std=("pchembl_std", "median"),
            measurement_count=("measurement_count", "sum"),
            assay_count=("assay_count", "sum"),
            document_count=("document_count", "sum"),
            molecule_count=("molecule_chembl_id", "nunique"),
            molecule_chembl_id=("molecule_chembl_id", "first"),
            smiles=("canonical_smiles", "first"),
            first_document_year=("first_document_year", "min"),
            latest_document_year=("latest_document_year", "max"),
        )
        .reset_index(drop=False)
    )

    grouped = grouped[
        ~(
            (grouped["max_pchembl"] >= ACTIVE_PCHEMBL_THRESHOLD)
            & (grouped["min_pchembl"] <= INACTIVE_PCHEMBL_THRESHOLD)
        )
    ].copy()
    grouped["label"] = grouped["median_pchembl"].apply(label_from_pchembl)
    grouped["compound_name"] = grouped["molecule_chembl_id"]
    grouped["scaffold"] = grouped["canonical_smiles"].apply(scaffold_from_smiles)
    grouped = grouped.sort_values("median_pchembl", ascending=False).reset_index(drop=True)
    grouped["dataset_index"] = np.arange(len(grouped), dtype=int)
    return grouped[
        [
            "dataset_index",
            "compound_name",
            "molecule_chembl_id",
            "canonical_smiles",
            "smiles",
            "scaffold",
            "median_activity_nM",
            "min_activity_nM",
            "max_activity_nM",
            "median_pchembl",
            "min_pchembl",
            "max_pchembl",
            "pchembl_std",
            "measurement_count",
            "assay_count",
            "document_count",
            "molecule_count",
            "first_document_year",
            "latest_document_year",
            "label",
        ]
    ]


def load_or_build_dataset(logger: RunLogger) -> pd.DataFrame:
    if DATASET_CACHE_PATH.exists():
        logger.log(f"Loading cached curated dataset from {DATASET_CACHE_PATH.name}")
        dataset_df = pd.read_csv(DATASET_CACHE_PATH)
    else:
        raw_df, downloaded_raw = load_or_download_raw_activities(logger)
        with logger.section("Building curated molecular dataset"):
            dataset_df = build_curated_dataset(raw_df, logger, allow_document_fetch=downloaded_raw or DOCUMENT_CACHE_PATH.exists())
        dataset_df.to_csv(DATASET_CACHE_PATH, index=False)

    if "dataset_index" not in dataset_df.columns:
        dataset_df = dataset_df.reset_index(drop=True).copy()
        dataset_df["dataset_index"] = np.arange(len(dataset_df), dtype=int)
        dataset_df.to_csv(DATASET_CACHE_PATH, index=False)

    dataset_df["dataset_index"] = pd.to_numeric(dataset_df["dataset_index"], errors="coerce").astype(int)
    return dataset_df.sort_values("dataset_index").reset_index(drop=True)


def build_classification_dataset(df: pd.DataFrame, logger: RunLogger) -> pd.DataFrame:
    classification_df = df.dropna(subset=["label"]).copy()
    classification_df["label"] = classification_df["label"].astype(int)
    logger.log(
        f"Classification dataset assembled with {len(classification_df)} molecules "
        f"({int(classification_df['label'].sum())} actives, {int((classification_df['label'] == 0).sum())} inactives)."
    )
    return classification_df.reset_index(drop=True)


def build_regression_dataset(df: pd.DataFrame, logger: RunLogger) -> pd.DataFrame:
    regression_df = df[np.isfinite(df["median_pchembl"])].copy()
    logger.log(
        f"Regression dataset assembled with {len(regression_df)} molecules spanning "
        f"pChEMBL {regression_df['median_pchembl'].min():.2f} to {regression_df['median_pchembl'].max():.2f}."
    )
    return regression_df.reset_index(drop=True)


def scaffold_groups(df: pd.DataFrame) -> np.ndarray:
    return df["scaffold"].fillna(df["canonical_smiles"]).astype(str).to_numpy()


def scaffold_split_integrity_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    train_ids = set(train_df["dataset_index"].astype(int).tolist())
    test_ids = set(test_df["dataset_index"].astype(int).tolist())
    train_smiles = set(train_df["canonical_smiles"].astype(str).tolist())
    test_smiles = set(test_df["canonical_smiles"].astype(str).tolist())
    train_scaffolds = set(scaffold_groups(train_df).tolist())
    test_scaffolds = set(scaffold_groups(test_df).tolist())
    return {
        "dataset_index_overlap_count": int(len(train_ids & test_ids)),
        "canonical_smiles_overlap_count": int(len(train_smiles & test_smiles)),
        "scaffold_overlap_count": int(len(train_scaffolds & test_scaffolds)),
    }


def split_classification_dataset(df: pd.DataFrame, logger: RunLogger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=int(round(1.0 / TEST_SIZE)), shuffle=True, random_state=RANDOM_SEED)
    groups = scaffold_groups(df)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    integrity = scaffold_split_integrity_summary(train_df, test_df)
    if any(int(value) > 0 for value in integrity.values()):
        raise RuntimeError(f"Scaffold split integrity violation detected: {integrity}")
    train_df.to_csv(CLASSIFICATION_TRAIN_PATH, index=False)
    test_df.to_csv(CLASSIFICATION_TEST_PATH, index=False)
    logger.log(
        f"Scaffold classification split: train={len(train_df)} ({train_df['label'].mean():.3f} positive), "
        f"test={len(test_df)} ({test_df['label'].mean():.3f} positive)."
    )
    logger.log(
        "Scaffold split integrity validated: "
        f"dataset_overlap={integrity['dataset_index_overlap_count']}, "
        f"smiles_overlap={integrity['canonical_smiles_overlap_count']}, "
        f"scaffold_overlap={integrity['scaffold_overlap_count']}."
    )
    return train_df, test_df


def split_regression_dataset(df: pd.DataFrame, logger: RunLogger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    groups = scaffold_groups(df)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    train_df.to_csv(REGRESSION_TRAIN_PATH, index=False)
    test_df.to_csv(REGRESSION_TEST_PATH, index=False)
    logger.log(f"Scaffold regression split: train={len(train_df)} molecules, test={len(test_df)} molecules.")
    return train_df, test_df


def build_temporal_split(
    df: pd.DataFrame,
    path_train: Path,
    path_test: Path,
    logger: RunLogger,
    require_binary_labels: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]] | None:
    temporal_df = df.dropna(subset=["first_document_year"]).copy()
    if temporal_df.empty:
        logger.log("Temporal split skipped because no publication years are available.")
        return None

    temporal_df["first_document_year"] = pd.to_numeric(temporal_df["first_document_year"], errors="coerce")
    temporal_df = temporal_df.dropna(subset=["first_document_year"]).copy()
    temporal_df["first_document_year"] = temporal_df["first_document_year"].astype(int)
    if temporal_df["first_document_year"].nunique() < 3:
        logger.log("Temporal split skipped because fewer than three distinct publication years are available.")
        return None

    year_counts = temporal_df["first_document_year"].value_counts().sort_index()
    total_rows = len(temporal_df)
    selected_years: List[int] = []
    running_rows = 0
    for year in sorted(year_counts.index, reverse=True):
        selected_years.append(int(year))
        running_rows += int(year_counts.loc[year])
        if running_rows / total_rows >= TEST_SIZE:
            break

    test_df = temporal_df[temporal_df["first_document_year"].isin(selected_years)].copy().reset_index(drop=True)
    train_df = temporal_df[~temporal_df["first_document_year"].isin(selected_years)].copy().reset_index(drop=True)
    if len(train_df) < 100 or len(test_df) < 50:
        logger.log("Temporal split skipped because the resulting train/test partitions are too small.")
        return None

    if require_binary_labels and (train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2):
        logger.log("Temporal split skipped because one side of the split does not contain both classes.")
        return None

    train_df.to_csv(path_train, index=False)
    test_df.to_csv(path_test, index=False)
    metadata = {
        "train_max_year": int(train_df["first_document_year"].max()),
        "test_years": sorted(selected_years),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }
    logger.log(
        f"Temporal split built with train years <= {metadata['train_max_year']} and test years {metadata['test_years']}."
    )
    return train_df, test_df, metadata


def save_classification_split_manifest(split_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    manifest_df = pd.DataFrame(split_rows).sort_values(["split_family", "split_label", "dataset_index"]).reset_index(drop=True)
    manifest_df.to_csv(CLASSIFICATION_SPLIT_MANIFEST_PATH, index=False)
    return manifest_df


def build_classification_splits(df: pd.DataFrame, logger: RunLogger) -> Dict[str, Any]:
    scaffold_train_df, scaffold_test_df = split_classification_dataset(df, logger)
    temporal_split = build_temporal_split(
        df,
        TEMPORAL_CLASSIFICATION_TRAIN_PATH,
        TEMPORAL_CLASSIFICATION_TEST_PATH,
        logger,
        require_binary_labels=True,
    )

    split_rows: List[Dict[str, Any]] = []
    for dataset_index in scaffold_train_df["dataset_index"].tolist():
        split_rows.append({"dataset_index": int(dataset_index), "split_family": "scaffold", "split_label": "train"})
        split_rows.append({"dataset_index": int(dataset_index), "split_family": "nested_external", "split_label": "internal_train"})
    for dataset_index in scaffold_test_df["dataset_index"].tolist():
        split_rows.append({"dataset_index": int(dataset_index), "split_family": "scaffold", "split_label": "holdout"})
        split_rows.append({"dataset_index": int(dataset_index), "split_family": "nested_external", "split_label": "external_validation"})

    internal_train_df = scaffold_train_df.copy().reset_index(drop=True)
    external_df = scaffold_test_df.copy().reset_index(drop=True)
    integrity = scaffold_split_integrity_summary(internal_train_df, external_df)
    metadata = {
        "strategy": "scaffold_external",
        "train_rows": len(internal_train_df),
        "test_rows": len(external_df),
        "train_positive_rate": float(internal_train_df["label"].mean()),
        "test_positive_rate": float(external_df["label"].mean()),
        "integrity": integrity,
    }
    if temporal_split is not None:
        temporal_train_df, temporal_test_df, temporal_metadata = temporal_split
        metadata["temporal_supplementary"] = temporal_metadata
        for dataset_index in temporal_train_df["dataset_index"].tolist():
            split_rows.append({"dataset_index": int(dataset_index), "split_family": "temporal", "split_label": "train"})
        for dataset_index in temporal_test_df["dataset_index"].tolist():
            split_rows.append({"dataset_index": int(dataset_index), "split_family": "temporal", "split_label": "holdout"})
    else:
        metadata["temporal_supplementary"] = None
    external_strategy = "scaffold_external"

    internal_train_df.to_csv(CLASSIFICATION_INTERNAL_TRAIN_PATH, index=False)
    external_df.to_csv(CLASSIFICATION_EXTERNAL_PATH, index=False)
    manifest_df = save_classification_split_manifest(split_rows)
    logger.log(
        f"Nested/external evaluation split prepared with strategy={external_strategy}, "
        f"internal_train={len(internal_train_df)}, external_validation={len(external_df)}."
    )
    return {
        "scaffold_train": scaffold_train_df,
        "scaffold_test": scaffold_test_df,
        "temporal_split": temporal_split,
        "internal_train": internal_train_df,
        "external_validation": external_df,
        "external_strategy": external_strategy,
        "external_metadata": metadata,
        "manifest": manifest_df,
    }


def build_regression_temporal_split(df: pd.DataFrame, logger: RunLogger) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]] | None:
    return build_temporal_split(
        df,
        TEMPORAL_REGRESSION_TRAIN_PATH,
        TEMPORAL_REGRESSION_TEST_PATH,
        logger,
        require_binary_labels=False,
    )
