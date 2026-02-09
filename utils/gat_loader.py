"""
GAT Checkpoint Loading for Mode 3.

This module handles loading GAT (Graph Attention Network) checkpoints
and generating predictions for the GAT + LLM Ensemble mode.

Supports both:
- GAT v1: nn.Embedding for MoA, cell line, compound
- GAT v2: KG-based embeddings for MoA/cell line (no compound)

PRIORITY: Use saved tokenizers from checkpoints when available.
This ensures 100% consistent predictions with training.
"""

import csv
import glob
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from baselines.run_gat_v2 import KnowledgeGraphBuilder as GATKnowledgeGraphBuilder

# Import GAT components (both v1 and v2)
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    
    # Import v2 model
    from baselines.run_gat_v2 import (
        KnowledgeGraphBuilder as GATKnowledgeGraphBuilder,
        GAT as GATModelV2,
    )
    
    # Import v1 model
    from baselines.run_gat import GAT as GATModelV1

    GAT_AVAILABLE = True
except ImportError as e:
    GAT_AVAILABLE = False
    GATKnowledgeGraphBuilder = None
    GATModelV1 = None
    GATModelV2 = None

logger = logging.getLogger(__name__)


# Default checkpoint directory for GAT v2
DEFAULT_GAT_CHECKPOINT_DIR = "checkpoints/260124_gat_v2"


class GATArgs:
    """Arguments container for GAT model configuration.
    
    Supports both v1 and v2 models:
    - v1: Uses nn.Embedding for MoA, cell line, compound
    - v2: Uses KG-based embeddings for MoA/cell line (no compound)
    """

    def __init__(self, saved_args: Dict = None):
        saved_args = saved_args or {}
        self.embed_dim = saved_args.get("embed_dim", 256)
        self.ffn_embed_dim = saved_args.get("ffn_embed_dim", 1024)
        self.num_layers = saved_args.get("num_layers", 4)
        self.dropout = saved_args.get("dropout", 0.1)
        self.batch_size = saved_args.get("batch_size", 512)
        self.num_workers = saved_args.get("num_workers", 4)
        self.val_proportion = saved_args.get("val_proportion", 0.1)
        self.vocab_size = saved_args.get("vocab_size")
        self.num_kgs = saved_args.get("num_kgs")
        self.num_cell_lines = saved_args.get("num_cell_lines")
        self.num_moas = saved_args.get("num_moas")
        self.num_compounds = saved_args.get("num_compounds")  # v1 only


class GATInferenceDataset(Dataset):
    """
    Dataset for GAT inference using tokenizers from checkpoint.
    
    This ensures 100% consistent token indices with the trained model.
    Supports both v1 (with compound) and v2 (no compound).
    """
    
    def __init__(
        self,
        test_csv: str,
        tokenizer: Dict[str, int],
        cell_line_tokenizer: Dict[str, int],
        moa_tokenizer: Dict[str, int] = None,
        compound_tokenizer: Dict[str, int] = None,  # v1 only
        is_v2: bool = False,
    ):
        """
        Initialize the inference dataset with pre-saved tokenizers.
        
        Args:
            test_csv: Path to test CSV file
            tokenizer: Entity tokenizer from checkpoint (gene -> index)
            cell_line_tokenizer: Cell line tokenizer from checkpoint
            moa_tokenizer: MoA tokenizer from checkpoint
            compound_tokenizer: Compound tokenizer from checkpoint (v1 only)
            is_v2: Whether this is for v2 model (no compound tokenization)
        """
        self.tokenizer = tokenizer
        self.cell_line_tokenizer = cell_line_tokenizer
        self.moa_tokenizer = moa_tokenizer or {}
        self.compound_tokenizer = compound_tokenizer or {}
        self.is_v2 = is_v2
        self.data = []
        
        # Load test data
        with open(test_csv, 'r') as f:
            reader = csv.DictReader(f)
            for item in reader:
                # Extract perturbation entity (compound name from pert string)
                pert_raw = item.get('pert', '')
                pert_entity = self._extract_pert_entity(pert_raw)
                
                gene_raw = item.get('gene', '')
                cell_line = item.get('cell_line', '')
                label = int(item.get('label', 0))
                
                # Get candidate_moa from CSV or parse from pert
                candidate_moa = item.get('candidate_moa', '')
                if not candidate_moa:
                    # Parse from pert string: "MEK inhibitor(AS-703026)" -> "MEK inhibitor"
                    candidate_moa = self._parse_candidate_moa(pert_raw)
                compound = item.get('compound', pert_entity)
                
                # Tokenize using saved tokenizers
                pert_idx = self.tokenizer.get(pert_entity, 0)  # v1 uses pert
                gene_idx = self.tokenizer.get(gene_raw, 0)
                cell_line_idx = self.cell_line_tokenizer.get(cell_line, 0)
                moa_idx = self.moa_tokenizer.get(candidate_moa, 0)
                compound_idx = self.compound_tokenizer.get(compound, 0) if not is_v2 else 0
                
                data_item = {
                    'gene': gene_idx,
                    'cell_line_idx': cell_line_idx,
                    'moa': moa_idx,
                    'label': label,
                    'pert_raw': pert_raw,
                    'gene_raw': gene_raw,
                    'cell_line': cell_line,
                    'gt_moa': item.get('gt_moa', ''),
                    'candidate_moa': candidate_moa,
                    'compound_raw': compound,
                }
                
                # v1 specific fields
                if not is_v2:
                    data_item['pert'] = pert_idx
                    data_item['compound'] = compound_idx
                
                self.data.append(data_item)
        
        logger.info(f"    Loaded {len(self.data)} samples using checkpoint tokenizer")
    
    def _extract_pert_entity(self, pert: str) -> str:
        """Extract perturbation entity (compound) from pert string."""
        if '(' in pert and ')' in pert:
            start = pert.index('(') + 1
            end = pert.index(')')
            return pert[start:end]
        return pert
    
    def _parse_candidate_moa(self, pert: str) -> str:
        """Parse candidate_moa from pert string."""
        if '(' in pert:
            return pert[:pert.index('(')].strip()
        return pert
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def gat_inference_collate_fn(batch, is_v2: bool = False):
    """Custom collate function for GAT inference.
    
    Supports both v1 (with pert, compound) and v2 (no compound tokenization).
    """
    # v1 includes 'pert' and 'compound', v2 does not
    if is_v2:
        keys = ['key', 'gene', 'label', 'pert_raw', 'gene_raw', 
                'gt_moa', 'candidate_moa', 'cell_line', 'compound_raw', 
                'cell_line_idx', 'moa']
        tensor_keys = ['gene', 'label', 'cell_line_idx', 'moa']
    else:
        keys = ['key', 'pert', 'gene', 'label', 'pert_raw', 'gene_raw', 
                'gt_moa', 'candidate_moa', 'cell_line', 'compound', 'compound_raw', 
                'cell_line_idx', 'moa']
        tensor_keys = ['pert', 'gene', 'label', 'cell_line_idx', 'moa', 'compound']
    
    result = {}
    
    for key in keys:
        values = [item.get(key) for item in batch if key in item]
        if not values:
            continue
        
        if key in tensor_keys:
            result[key] = torch.tensor(values, dtype=torch.long)
        else:
            result[key] = values
    
    return result


def make_collate_fn(is_v2: bool):
    """Factory function to create collate function for specific version."""
    def collate_fn(batch):
        return gat_inference_collate_fn(batch, is_v2=is_v2)
    return collate_fn


def load_gat_predictions_for_organ(
    seed: int,
    organ: str,
    gat_checkpoint_dir: str,
    train_csv: str,
    test_csv: str,
    kg_dir: str,
    kg_sources: List[str],
    task_type: str,
    device: str = "cuda",
    kg_builder: Optional["GATKnowledgeGraphBuilder"] = None,
    return_failed_flag: bool = False,
) -> Dict[Tuple[str, str, str], float]:
    """
    Load GAT predictions for a single organ.
    
    Automatically detects checkpoint version and loads appropriate model:
    - v1: Uses nn.Embedding for MoA, cell line, compound (MLP input = 4*embed_dim)
    - v2: Uses KG-based embeddings for MoA/cell line, no compound (MLP input = 3*embed_dim)
    
    PRIORITY ORDER for tokenizer:
    1. Use tokenizer/cell_line_tokenizer/moa_tokenizer saved in checkpoint (100% consistent)
    2. Fallback: Return empty dict or 0.5 fallback if return_failed_flag=True

    Args:
        seed: Random seed used in the GAT run (default: 0)
        organ: Organ name in lowercase format (e.g., "skin", "colon")
        gat_checkpoint_dir: Directory containing GAT checkpoints
        train_csv: Path to training CSV (not used when tokenizer is in checkpoint)
        test_csv: Path to test CSV
        kg_dir: Directory containing knowledge graph data (not used when tokenizer is in checkpoint)
        kg_sources: List of knowledge graph sources (not used when tokenizer is in checkpoint)
        task_type: Task type (task1 or task2)
        device: Device to use for inference
        kg_builder: Optional pre-built knowledge graph builder (not used when tokenizer is in checkpoint)
        return_failed_flag: If True, return tuple (predictions, gat_failed) instead of just predictions

    Returns:
        Dictionary mapping (pert, gene, cell_line) tuples to prediction probabilities.
        If return_failed_flag=True, returns (predictions_dict, gat_failed_bool)
    """
    gat_failed = False
    
    if not GAT_AVAILABLE:
        logger.error("GAT components not available.")
        gat_failed = True
        if return_failed_flag:
            return {}, gat_failed
        return {}

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # GAT models are trained only on task1, so always load task1 checkpoint
    # regardless of the task_type being processed
    checkpoint_path = os.path.join(
        gat_checkpoint_dir, "task1", organ, f"best_model_{seed}.pt"
    )

    if not os.path.exists(checkpoint_path):
        logger.warning(
            f"  GAT checkpoint not found for organ {organ}: {checkpoint_path}"
        )
        gat_failed = True
        if return_failed_flag:
            return {}, gat_failed
        return {}

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    except Exception as e:
        logger.error(f"  Failed to load GAT checkpoint for organ {organ}: {e}")
        gat_failed = True
        if return_failed_flag:
            return {}, gat_failed
        return {}

    # Detect checkpoint version by checking for v1-specific keys
    state_dict = checkpoint.get("model_state_dict", {})
    is_v1_checkpoint = "compound_embedding.weight" in state_dict or "moa_embedding.weight" in state_dict
    is_v2_checkpoint = "moa_projection.0.weight" in state_dict or "moa_fallback_embedding.weight" in state_dict
    
    if is_v1_checkpoint:
        logger.info(f"    Detected GAT v1 checkpoint for organ {organ}")
    elif is_v2_checkpoint:
        logger.info(f"    Detected GAT v2 checkpoint for organ {organ}")
    else:
        logger.warning(f"    Unknown checkpoint version for organ {organ}, assuming v1")
        is_v1_checkpoint = True

    # Get saved args from checkpoint
    saved_args = checkpoint.get("args", {})
    args = GATArgs(saved_args)
    
    # CRITICAL: Check if tokenizer is saved in checkpoint
    saved_tokenizer = checkpoint.get("tokenizer")
    saved_cell_line_tokenizer = checkpoint.get("cell_line_tokenizer")
    saved_moa_tokenizer = checkpoint.get("moa_tokenizer")
    saved_compound_tokenizer = checkpoint.get("compound_tokenizer") if is_v1_checkpoint else None
    
    if saved_tokenizer is None:
        logger.error(
            f"  ERROR: Checkpoint for organ '{organ}' does not contain tokenizer.\n"
            f"  This checkpoint is invalid or corrupted.\n"
            f"  Please re-train the GAT model to create a new checkpoint."
        )
        gat_failed = True
        if return_failed_flag:
            return {}, gat_failed
        return {}
    
    if saved_cell_line_tokenizer is None:
        logger.warning(
            f"  Checkpoint for organ {organ} missing cell_line_tokenizer. Using default."
        )
        saved_cell_line_tokenizer = {None: 0, '': 0}
    
    if saved_moa_tokenizer is None:
        logger.warning(
            f"  Checkpoint for organ {organ} missing moa_tokenizer. Using default."
        )
        saved_moa_tokenizer = {}
    
    if is_v1_checkpoint and saved_compound_tokenizer is None:
        logger.warning(
            f"  v1 checkpoint for organ {organ} missing compound_tokenizer. Using default."
        )
        saved_compound_tokenizer = {}
    
    # Log tokenizer info
    if is_v1_checkpoint:
        logger.info(
            f"    Using GAT v1 saved tokenizer from checkpoint "
            f"(vocab={len(saved_tokenizer)}, cell_lines={len(saved_cell_line_tokenizer)}, "
            f"moas={len(saved_moa_tokenizer)}, compounds={len(saved_compound_tokenizer or {})})"
        )
    else:
        # GAT v2 specific: Load vocab, moa_vocab, cell_line_vocab, moa_target_mapping, cell_type_genes
        saved_vocab = checkpoint.get("vocab", saved_tokenizer)
        saved_moa_vocab = checkpoint.get("moa_vocab", saved_moa_tokenizer)
        saved_cell_line_vocab = checkpoint.get("cell_line_vocab", saved_cell_line_tokenizer)
        saved_moa_target_mapping = checkpoint.get("moa_target_mapping", {})
        saved_cell_type_genes = checkpoint.get("cell_type_genes", {})
        
        logger.info(
            f"    Using GAT v2 saved tokenizer from checkpoint "
            f"(vocab={len(saved_tokenizer)}, cell_lines={len(saved_cell_line_tokenizer)}, "
            f"moas={len(saved_moa_tokenizer)})"
        )
    
    # Create inference dataset using saved tokenizers
    if not os.path.exists(test_csv):
        logger.warning(f"  Test CSV not found: {test_csv}")
        gat_failed = True
        if return_failed_flag:
            return {}, gat_failed
        return {}
    
    try:
        dataset = GATInferenceDataset(
            test_csv=test_csv,
            tokenizer=saved_tokenizer,
            cell_line_tokenizer=saved_cell_line_tokenizer,
            moa_tokenizer=saved_moa_tokenizer,
            compound_tokenizer=saved_compound_tokenizer if is_v1_checkpoint else None,
            is_v2=not is_v1_checkpoint,
        )
    except Exception as e:
        logger.error(f"  Failed to create dataset for organ {organ}: {e}")
        gat_failed = True
        if return_failed_flag:
            return {}, gat_failed
        return {}
    
    if len(dataset) == 0:
        logger.warning(f"  No test samples found for organ {organ}")
        gat_failed = True
        if return_failed_flag:
            return {}, gat_failed
        return {}
    
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=make_collate_fn(is_v2=not is_v1_checkpoint),
    )

    # Load model from checkpoint
    try:
        from torch_geometric.data import Data

        # Reconstruct graph from checkpoint state dict
        checkpoint_edge_index = state_dict["edge_index"]
        if checkpoint_edge_index.shape[0] == 2:
            checkpoint_edge_index = checkpoint_edge_index.T

        checkpoint_graph = Data(
            x=state_dict["x"],
            edge_index=checkpoint_edge_index,
            edge_attr=state_dict["edge_attr"],
        )

        # Get model dimensions from checkpoint args
        model_vocab_size = args.vocab_size or len(saved_tokenizer)
        model_num_kgs = args.num_kgs or 1
        model_num_cell_lines = args.num_cell_lines or len(saved_cell_line_tokenizer)
        model_num_moas = args.num_moas or len(saved_moa_tokenizer) or 1

        if is_v1_checkpoint:
            # GAT v1 model: Uses nn.Embedding for compound
            model_num_compounds = args.num_compounds or len(saved_compound_tokenizer or {}) or 1
            
            model = GATModelV1(
                vocab_size=model_vocab_size,
                num_kgs=model_num_kgs,
                embed_dim=args.embed_dim,
                ffn_embed_dim=args.ffn_embed_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                graph=checkpoint_graph,
                num_cell_lines=model_num_cell_lines,
                num_moas=model_num_moas,
                num_compounds=model_num_compounds,
            ).to(device_obj)
        else:
            # GAT v2 model: No num_compounds, uses KG-based embeddings
            model = GATModelV2(
                vocab_size=model_vocab_size,
                num_kgs=model_num_kgs,
                embed_dim=args.embed_dim,
                ffn_embed_dim=args.ffn_embed_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                graph=checkpoint_graph,
                num_cell_lines=model_num_cell_lines,
                num_moas=model_num_moas,
                tokenizer=saved_tokenizer,
                moa_target_mapping=saved_moa_target_mapping,
                cell_type_genes=None,  # Not used in v2 (MoA + gene only)
                moa_vocab=saved_moa_vocab,
                cell_line_vocab=saved_cell_line_vocab,
            ).to(device_obj)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    except Exception as e:
        logger.error(f"  Failed to initialize GAT model for organ {organ}: {e}")
        import traceback
        traceback.print_exc()
        gat_failed = True
        if return_failed_flag:
            return {}, gat_failed
        return {}

    # Run inference
    predictions = {}
    all_probs = []

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch_device = {
                    k: v.to(device_obj) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                # Debug first batch to verify MoA embeddings are different
                debug_moa = (batch_idx == 0)
                output = model(batch_device, debug_moa=debug_moa)
                probs = F.softmax(output, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().numpy().tolist())

                for i in range(len(probs)):
                    pert = batch["pert_raw"][i] if "pert_raw" in batch else ""
                    gene = batch["gene_raw"][i] if "gene_raw" in batch else ""
                    cell_line = batch["cell_line"][i] if "cell_line" in batch else ""
                    compound = batch["compound_raw"][i] if "compound_raw" in batch else ""
                    prob_value = float(probs[i].cpu().item())
                    # Store with pert-based key (primary)
                    predictions[(pert, gene, cell_line)] = prob_value
                    # Also store with compound-based key for sorted_pathway compatibility
                    if compound:
                        predictions[(compound, gene, cell_line)] = prob_value

        # Log statistics
        if all_probs:
            min_prob = min(all_probs)
            max_prob = max(all_probs)
            up_count = sum(1 for p in all_probs if p > 0.5)
            down_count = len(all_probs) - up_count
            version_str = "v1" if is_v1_checkpoint else "v2"
            logger.info(
                f"    Organ {organ}: {len(predictions)} GAT {version_str} predictions, "
                f"prob range [{min_prob:.3f}, {max_prob:.3f}], "
                f"up={up_count}, down={down_count}"
            )

    except Exception as e:
        version_str = "v1" if is_v1_checkpoint else "v2"
        logger.error(f"  GAT {version_str} inference failed for organ {organ}: {e}")
        import traceback
        traceback.print_exc()
        gat_failed = True
        if return_failed_flag:
            return {}, gat_failed
        return {}

    if return_failed_flag:
        return predictions, gat_failed
    return predictions


def load_all_gat_predictions(
    seed: int,
    df: pd.DataFrame,
    gat_checkpoint_dir: str,
    train_csv: str,
    test_csv: str,
    kg_dir: str,
    kg_sources: List[str],
    task_type: str,
    device: str = "cuda",
    organs: Optional[List[str]] = None,
) -> Dict[Tuple[str, str, str], float]:
    """
    Load GAT predictions for all organs in the dataframe.

    This function iterates over all unique organs in the dataframe and loads
    GAT predictions for each organ using its respective checkpoint.
    
    IMPORTANT: Requires checkpoints with saved tokenizers for consistent predictions.

    Args:
        seed: Random seed used in the GAT run (default: 0)
        df: DataFrame containing samples with 'organ' column
        gat_checkpoint_dir: Directory containing GAT checkpoints
        train_csv: Path to training CSV (base path for organ-specific CSVs)
        test_csv: Path to test CSV (base path for organ-specific CSVs)
        kg_dir: Directory containing knowledge graph data
        kg_sources: List of knowledge graph sources
        task_type: Task type (task1 or task2)
        device: Device to use for inference
        organs: Optional list of specific organs to load (if None, load all organs in df)

    Returns:
        Dictionary mapping (pert, gene, cell_line) tuples to prediction probabilities
    """
    if not GAT_AVAILABLE:
        logger.error("GAT components not available. Cannot load GAT predictions.")
        return {}

    if "organ" not in df.columns:
        logger.error(
            "DataFrame does not contain 'organ' column. Cannot load GAT checkpoint."
        )
        return {}

    # Get organs to process: use specified organs or all unique organs from df
    if organs is not None:
        # Normalize organ names for comparison
        organs_normalized = [o.lower().replace(" ", "_") for o in organs]
        unique_organs = [
            o for o in df["organ"].unique()
            if o.lower().replace(" ", "_") in organs_normalized
        ]
        logger.info(f"  Filtering to specified organs: {organs}")
    else:
        unique_organs = df["organ"].unique()
    logger.info(f"{'=' * 60}")
    logger.info(f"Loading GAT predictions for Mode 3")
    logger.info(f"  Task: {task_type}")
    logger.info(f"  Unique organs: {list(unique_organs)}")
    logger.info(f"  Using saved tokenizers from checkpoints (100% consistent)")
    logger.info(f"{'=' * 60}")

    all_predictions = {}

    for organ_raw in unique_organs:
        organ = organ_raw.lower().replace(" ", "_")

        # Determine organ-specific train/test CSV paths
        # For case_study, use the provided path directly (no organ-specific CSV)
        if "case_study" in test_csv.lower():
            organ_train_csv = train_csv
            organ_test_csv = test_csv
        # Skip organ-specific path if using integrated CSV (contains all organs)
        elif "integrated" in train_csv.lower() or "integrated" in test_csv.lower():
            organ_train_csv = train_csv
            organ_test_csv = test_csv
        elif "organs" in train_csv:
            organ_train_csv = train_csv.replace(
                train_csv.split("/")[-1].split("_")[0], organ
            )
            organ_test_csv = test_csv.replace(
                test_csv.split("/")[-1].split("_")[0], organ
            )
        else:
            base_dir = os.path.dirname(train_csv)
            organ_train_csv = os.path.join(
                base_dir, "organs", task_type, f"{organ}_train.csv"
            )
            organ_test_csv = os.path.join(
                base_dir, "organs", task_type, f"{organ}_test.csv"
            )

        # Fallback to original paths if organ-specific doesn't exist
        if not os.path.exists(organ_train_csv):
            organ_train_csv = train_csv
        if not os.path.exists(organ_test_csv):
            organ_test_csv = test_csv

        logger.info(f"  Loading predictions for organ: {organ}")
        logger.info(f"    Using test CSV: {organ_test_csv}")

        organ_predictions = load_gat_predictions_for_organ(
            seed=seed,
            organ=organ,
            gat_checkpoint_dir=gat_checkpoint_dir,
            train_csv=organ_train_csv,
            test_csv=organ_test_csv,
            kg_dir=kg_dir,
            kg_sources=kg_sources,
            task_type=task_type,
            device=device,
            kg_builder=None,
        )

        all_predictions.update(organ_predictions)

    logger.info(f"  Total GAT predictions loaded: {len(all_predictions)}")

    return all_predictions


def load_precomputed_gat_predictions(
    df: pd.DataFrame,
    precompute_gat_dir: str,
    task_type: str,
    seed: int = 0,
) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """
    Load precomputed GAT predictions from saved JSON files.
    
    This function loads predictions from JSON files saved by a previous GAT run,
    instead of loading the GAT checkpoint and running inference.
    
    File path pattern (new format):
        {precompute_gat_dir}/{organ}/{task_type}/{test_case_id}/{gt_moa}/{compound}/{candidate_moa}_{seed}.json
    
    Also supports legacy format:
        {precompute_gat_dir}/{organ}/temp_0.0/{task_type}/json/{gt_moa}/{gt_moa}{compound}_{seed}.json
    
    Args:
        df: DataFrame containing samples with 'organ', 'gt_moa', 'compound' columns
        precompute_gat_dir: Directory containing precomputed GAT results
                           (e.g., results/260127_gat_all_organs/GAT)
        task_type: Task type (task1 or task2)
        seed: Random seed used in the original GAT run (default: 0)
    
    Returns:
        Dictionary mapping (pert, gene, cell_line) tuples to dict with:
            - 'prob': prediction probability (predicted_prob)
            - 'label': predicted label (parsed_prediction as int)
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"Loading PRECOMPUTED GAT predictions for Mode 3")
    logger.info(f"  Task: {task_type}")
    logger.info(f"  Precompute dir: {precompute_gat_dir}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"{'=' * 60}")
    
    if "organ" not in df.columns:
        logger.error("DataFrame does not contain 'organ' column.")
        return {}
    
    all_predictions = {}
    loaded_files = 0
    skipped_files = 0
    
    # Get unique organs
    unique_organs = df["organ"].unique()
    logger.info(f"  Unique organs: {list(unique_organs)}")
    
    for organ_raw in unique_organs:
        organ = organ_raw.lower().replace(" ", "_")
        
        # Try new path format first: {precompute_gat_dir}/{organ}/{task_type}/...
        organ_dir = os.path.join(precompute_gat_dir, organ, task_type)
        
        # Fallback to legacy format: {precompute_gat_dir}/{organ}/temp_0.0/{task_type}/json/...
        if not os.path.exists(organ_dir):
            organ_dir = os.path.join(precompute_gat_dir, organ, "temp_0.0", task_type, "json")
        
        if not os.path.exists(organ_dir):
            logger.warning(f"  Precomputed dir not found for organ {organ}: {organ_dir}")
            continue
        
        # Find all JSON files in the organ directory (excluding metrics files)
        json_pattern = os.path.join(organ_dir, "**", f"*_{seed}.json")
        json_files = [f for f in glob.glob(json_pattern, recursive=True) if "metrics" not in f]
        
        logger.info(f"  Organ {organ}: found {len(json_files)} JSON files (seed={seed})")
        
        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                metadata = data.get("metadata", {})
                samples = data.get("samples", [])
                
                # Get pert from metadata
                pert = metadata.get("pert", "")
                if not pert:
                    logger.warning(f"  No 'pert' in metadata: {json_path}")
                    skipped_files += 1
                    continue
                
                for sample in samples:
                    gene = sample.get("gene", "")
                    cell_line = sample.get("cell_line", "")
                    predicted_prob = sample.get("predicted_prob")
                    parsed_prediction = sample.get("parsed_prediction")
                    
                    if gene and predicted_prob is not None:
                        key = (pert, gene, cell_line)
                        all_predictions[key] = {
                            "prob": float(predicted_prob),
                            "label": int(parsed_prediction) if parsed_prediction is not None else (1 if predicted_prob > 0.5 else 0),
                        }
                
                loaded_files += 1
                
            except Exception as e:
                logger.warning(f"  Failed to load {json_path}: {e}")
                skipped_files += 1
                continue
    
    logger.info(f"  Loaded {loaded_files} JSON files, skipped {skipped_files}")
    logger.info(f"  Total precomputed GAT predictions: {len(all_predictions)}")
    
    # Log sample statistics
    if all_predictions:
        probs = [v["prob"] for v in all_predictions.values()]
        labels = [v["label"] for v in all_predictions.values()]
        up_count = sum(1 for l in labels if l == 1)
        down_count = len(labels) - up_count
        logger.info(
            f"  Prob range: [{min(probs):.4f}, {max(probs):.4f}], "
            f"up={up_count}, down={down_count}"
        )
    
    return all_predictions
