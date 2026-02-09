"""
Data loading utilities for compound perturbation benchmarking.

Loads CSV datasets with schema:
pert, gene, label, split, candidate_moa, gt_moa, cell_line, compound
(selection_tier is optional)
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def load_benchmark_csv(csv_path: str, split: str = "test") -> pd.DataFrame:
    """
    Load benchmark CSV dataset.

    Args:
        csv_path: Path to CSV file
        split: Dataset split to filter (e.g., 'test', 'train')

    Returns:
        DataFrame with benchmark data
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate required columns (selection_tier is optional)
    required_cols = ['pert', 'gene', 'label', 'split',
                     'candidate_moa', 'gt_moa', 'cell_line', 'compound']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter by split if specified
    if split:
        df = df[df['split'] == split].copy()

    logger.info(f"Loaded {len(df)} samples from {csv_path} (split={split})")
    logger.info(f"  Cell lines: {df['cell_line'].unique().tolist()}")
    logger.info(f"  Unique perturbations: {df['pert'].nunique()}")
    logger.info(f"  Unique genes: {df['gene'].nunique()}")

    return df


def group_by_cell_line(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group dataframe by cell line.

    Args:
        df: Benchmark dataframe

    Returns:
        Dictionary mapping cell_line -> DataFrame
    """
    grouped = {}
    for cell_line in df['cell_line'].unique():
        grouped[cell_line] = df[df['cell_line'] == cell_line].copy()
        logger.info(f"  {cell_line}: {len(grouped[cell_line])} samples")

    return grouped


def get_cell_line_stats(df: pd.DataFrame) -> Dict:
    """
    Get statistics for a cell line dataset.

    Args:
        df: DataFrame for a single cell line

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_samples': len(df),
        'num_perturbations': df['pert'].nunique(),
        'num_genes': df['gene'].nunique(),
        'num_gt_moas': df['gt_moa'].nunique(),
        'label_distribution': df['label'].value_counts().to_dict(),
        'gt_moas': df['gt_moa'].unique().tolist()
    }
    return stats
