#!/usr/bin/env python3
"""
Unified Multi-Scientist Multi-Judge Pipeline for LINCSQA.

This script integrates both task types with configurable options:
- Task1: Gene regulation direction prediction (with GAT, answers: upregulated/downregulated)
- Task2: MoA prediction case study (without GAT, answers: yes/no or yes/no/uncertain)

Key Features:
- Modular pipeline components from utils/pipeline/
- Configurable GAT usage (--use_gat / --no_gat)
- Configurable uncertain answer mode for task2 (--allow_uncertain)
- Support for sorted pathway JSON data
- Full traceability with saved prompts and reasoning

Usage Examples:
    # Task1 with GAT (default)
    python run.py --task_type task1 --use_gat --organs blood \\
        --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B

    # Task2 without GAT, with uncertain answers
    python run.py --task_type task2 --no_gat --allow_uncertain \\
        --case_study case1_braf --organs blood \\
        --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B

    # Task2 without uncertain answers (yes/no only)
    python run.py --task_type task2 --no_gat \\
        --case_study case2_kras --organs blood
"""

import argparse
import logging
import os
import random
import sys
import time
from typing import Dict, List

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    VLLMClient,
    MultiScientistMultiJudgePipeline,
    # Data loaders
    SortedSampleInfo,
    load_sorted_pathway_data,
    CrossFileBatchIterator,
    load_benchmark_csv,
    # History
    HistoryManager,
    # Config
    SampleResult,
    load_domain_knowledge,
    # Result saver
    save_results_json,
    # Knowledge
    UnifiedKGContext,
    KnowledgeRetrievalLayer,
    # GAT
    load_all_gat_predictions,
    load_precomputed_gat_predictions,
    # Gene ordering
    StringDBLocal,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Global domain knowledge (loaded once at startup)
MOA_KNOWLEDGE: Dict[str, str] = {}
COMPOUND_KNOWLEDGE: Dict[str, str] = {}


def run_sorted_pathway_pipeline(
    samples_by_organ: Dict[str, Dict[str, List[SortedSampleInfo]]],
    organs: List[str],
    pipeline: MultiScientistMultiJudgePipeline,
    task_type: str,
    gat_predictions: Dict,
    args,
) -> List[SampleResult]:
    """
    Run Multi-Scientist Multi-Judge pipeline with sorted pathway data.

    Processes one organ at a time with cross-file batching (one sample
    from each active source JSON file per batch).
    
    Args:
        samples_by_organ: Dictionary mapping organ -> source_key -> samples.
        organs: List of organs to process.
        pipeline: MultiScientistMultiJudgePipeline instance.
        task_type: Task type ('task1' or 'task2').
        gat_predictions: Dictionary of precomputed GAT predictions.
        args: Command line arguments.
        
    Returns:
        List of SampleResult objects.
    """
    all_results = []
    
    for organ in organs:
        organ_sources = samples_by_organ.get(organ, {})
        if not organ_sources:
            logger.info(f"\nSkipping organ {organ}: no source files found")
            continue

        total_samples = sum(len(v) for v in organ_sources.values())
        batch_iterator = CrossFileBatchIterator(organ_sources, args.batch_size)
        total_batches = len(batch_iterator)

        logger.info(f"\nProcessing organ {organ}: {total_samples} samples across {len(organ_sources)} source files")
        logger.info(
            "Using cross-file batching: up to %d active sources per batch",
            min(args.batch_size, len(organ_sources)),
        )

        organ_results = []
        batch_idx = 0
        
        for batch_sample_infos in batch_iterator:
            batch_idx += 1
            batch_samples = [info.sample for info in batch_sample_infos]

            source_keys = [info.source_key for info in batch_sample_infos]
            logger.info(
                f"  Batch {batch_idx}/{total_batches} "
                f"({len(batch_samples)} samples from {len(set(source_keys))} sources)"
            )

            batch_results = pipeline.process_batch(batch_samples, gat_predictions, parallelize=True)
            organ_results.extend(batch_results)
            all_results.extend(batch_results)

            passed = sum(1 for r in organ_results if r.status == "passed")
            correct = sum(1 for r in organ_results if r.is_correct)
            logger.info(
                f"    Progress: {len(organ_results)}/{total_samples}, "
                f"Passed: {passed}, Accuracy: {100*correct/len(organ_results):.1f}%"
            )

        save_results_json(organ_results, args, task_type, organ)

    return all_results


def run_csv_pipeline(
    df: pd.DataFrame,
    pipeline: MultiScientistMultiJudgePipeline,
    task_type: str,
    gat_predictions: Dict,
    args,
) -> List[SampleResult]:
    """
    Run Multi-Scientist Multi-Judge pipeline with CSV data.
    
    Args:
        df: DataFrame with samples.
        pipeline: MultiScientistMultiJudgePipeline instance.
        task_type: Task type ('task1' or 'task2').
        gat_predictions: Dictionary of precomputed GAT predictions.
        args: Command line arguments.
        
    Returns:
        List of SampleResult objects.
    """
    all_results = []
    
    organs = df["organ"].unique() if "organ" in df.columns else ["unknown"]
    
    for organ in organs:
        organ_df = df[df["organ"] == organ] if "organ" in df.columns else df
        logger.info(f"\nProcessing organ: {organ} ({len(organ_df)} samples)")
        
        # Convert to list of dicts
        samples = []
        for _, row in organ_df.iterrows():
            samples.append({
                "pert": row["pert"],
                "gene": row["gene"],
                "cell_line": row["cell_line"],
                "label": row["label"],
                "organ": organ,
                "gt_moa": row.get("gt_moa", ""),
                "candidate_moa": row.get("candidate_moa", ""),
                "compound": row.get("compound", ""),
                "test_case_id": row.get("test_case_id", ""),
            })
        
        # Process in batches
        organ_results = []
        batch_size = args.batch_size
        total_batches = (len(samples) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]
            
            logger.info(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch_samples)} samples)")
            
            batch_results = pipeline.process_batch(batch_samples, gat_predictions)
            organ_results.extend(batch_results)
            
            # Progress
            passed = sum(1 for r in organ_results if r.status == "passed")
            correct = sum(1 for r in organ_results if r.is_correct)
            logger.info(f"    Progress: {len(organ_results)}/{len(samples)}, "
                       f"Passed: {passed}, Accuracy: {100*correct/len(organ_results):.1f}%")
        
        # Save organ results
        save_results_json(organ_results, args, task_type, organ)
        all_results.extend(organ_results)
    
    return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Multi-Scientist Multi-Judge Pipeline for LINCSQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Task1 with GAT (gene regulation direction prediction)
  python run.py --task_type task1 --use_gat --organs blood \\
      --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B

  # Task2 without GAT, allowing uncertain answers (MoA case study)
  python run.py --task_type task2 --no_gat --allow_uncertain \\
      --case_study case1_braf --organs blood

  # Task2 with yes/no only answers
  python run.py --task_type task2 --no_gat --case_study case2_kras --organs blood
        """
    )

    # Task configuration
    parser.add_argument("--task_type", type=str, required=True, choices=["task1", "task2"],
                        help="Task type: task1 (gene regulation direction) or task2 (MoA prediction)")
    parser.add_argument("--case_study", type=str, choices=["case1_braf", "case2_kras"],
                        help="Case study name for task2 (required for task2)")

    # GAT configuration
    gat_group = parser.add_mutually_exclusive_group()
    gat_group.add_argument("--use_gat", action="store_true", default=True,
                           help="Enable GAT predictions (default for task1)")
    gat_group.add_argument("--no_gat", action="store_true",
                           help="Disable GAT predictions (default for task2)")

    # Answer mode configuration
    parser.add_argument("--allow_uncertain", action="store_true",
                        help="Allow 'uncertain' answers for task2 (default: yes/no only)")

    # Data configuration
    parser.add_argument("--csv_path", type=str, default=None,
                        help="CSV path (used when not using sorted pathway mode)")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to use (default: test)")
    parser.add_argument("--use_sorted_pathway", action="store_true", default=True,
                        help="Use sorted pathway JSON files for data loading (default: True)")
    parser.add_argument("--sorted_pathway_dir", type=str, default=None,
                        help="Base directory for sorted pathway data (auto-detected if not set)")
    parser.add_argument("--organs", type=str, nargs="+", required=True,
                        help="List of organs to process (required)")

    # Model configuration
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Model name for inference")
    parser.add_argument("--vllm_port", type=int, default=None,
                        help="VLLM server port (default: from VLLM_PORT env or 8000)")

    # Inference settings
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (default: 0.6)")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum tokens to generate (default: 4096)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Maximum parallel workers for LLM calls (default: 16)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for processing (default: 2)")
    parser.add_argument("--max_retry", type=int, default=2,
                        help="Maximum retry attempts per sample (default: 2)")

    # Knowledge settings
    parser.add_argument("--max_pathways", type=int, default=10,
                        help="Maximum pathways to include in context (default: 10)")

    # History settings
    parser.add_argument("--use_history", action="store_true", default=False,
                        help="Use STRING DB similarity-based history for integration agent")
    parser.add_argument("--history_topk", type=int, default=3,
                        help="Number of top STRING-similar history genes to include with detailed context (default: 3)")

    # Ordering mode (for ablation study)
    parser.add_argument("--ordering_mode", type=str, default="default",
                        choices=["default", "random", "reverse"],
                        help="Sample ordering within each source file: "
                             "default (pre-sorted by combined_score), "
                             "random (shuffled), "
                             "reverse (hardest first)")

    # Output configuration
    parser.add_argument("--project_name", type=str, default="pbioagent_project",
                        help="Project name for output directory (auto-generated if not set)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Base results directory (default: results)")

    # Cell line filtering
    parser.add_argument("--cell_lines", type=str, nargs="+", default=None,
                        help="Filter to specific cell lines")

    # DepMap settings
    parser.add_argument("--use_depmap", action="store_true", default=True,
                        help="Enable DepMap context (default: True)")
    parser.add_argument("--no_depmap", action="store_true",
                        help="Disable DepMap context")
    parser.add_argument("--depmap_path", type=str, default="data/kg/depmap",
                        help="Path to DepMap data")

    # GAT checkpoint settings
    parser.add_argument("--gat_checkpoint_dir", type=str, default="checkpoints/260119_gat_all_organs",
                        help="Directory containing GAT model checkpoints")
    parser.add_argument("--gat_kg_dir", type=str, default="data/kg",
                        help="Directory containing knowledge graph data for GAT")
    parser.add_argument("--gat_kg_sources", type=str, nargs="+",
                        default=["SIGNOR", "GO", "Reactome", "STRING", "CORUM", "BioPlex"],
                        help="Knowledge graph sources for GAT")
    parser.add_argument("--gat_device", type=str, default="cpu",
                        help="Device for GAT inference (default: cpu)")
    parser.add_argument("--use_precompute_gat", action="store_true",
                        help="Use precomputed GAT predictions instead of inference")
    parser.add_argument("--precompute_gat_dir", type=str, default="results/260114_gat_all_organs/GAT",
                        help="Directory containing precomputed GAT predictions")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate and apply task-specific defaults
    task_type = args.task_type
    
    # Task2 requires case_study
    if task_type == "task2" and not args.case_study:
        logger.error("--case_study is required for task2")
        sys.exit(1)
    
    # Apply GAT defaults based on task type
    if args.no_gat:
        use_gat = False
    elif task_type == "task2":
        # Task2 defaults to no GAT unless explicitly enabled
        use_gat = False
        logger.info("Task2: GAT disabled by default (use --use_gat to enable)")
    else:
        # Task1 defaults to using GAT
        use_gat = True
    
    # Allow uncertain only valid for task2
    allow_uncertain = args.allow_uncertain and task_type == "task2"
    if args.allow_uncertain and task_type != "task2":
        logger.warning("--allow_uncertain is only valid for task2, ignoring")
    
    # Auto-detect sorted_pathway_dir based on task type
    if args.sorted_pathway_dir is None:
        if task_type == "task1":
            args.sorted_pathway_dir = "data/lincsqa/gene_regulation_dir_pred/combined_score"
        else:
            # task2 uses case_study subdirectory
            args.sorted_pathway_dir = f"data/lincsqa/case_study/sorted/combined_score"
    
    # Auto-generate project name if not provided
    if args.project_name is None:
        timestamp = time.strftime("%y%m%d")
        gat_suffix = "_gat" if use_gat else "_no_gat"
        uncertain_suffix = "_uncertain" if allow_uncertain else ""
        case_suffix = f"_{args.case_study}" if args.case_study else ""
        args.project_name = f"{timestamp}_ours_{task_type}{gat_suffix}{uncertain_suffix}{case_suffix}"

    # Log configuration
    logger.info("\n" + "=" * 80)
    logger.info("UNIFIED MULTI-SCIENTIST MULTI-JUDGE PIPELINE")
    logger.info("=" * 80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Task Type: {task_type}")
    if args.case_study:
        logger.info(f"  Case Study: {args.case_study}")
    logger.info(f"  GAT: {'enabled' if use_gat else 'disabled'}")
    logger.info(f"  Uncertain Answers: {'allowed' if allow_uncertain else 'not allowed (yes/no only)'}")
    logger.info(f"  Data Mode: {'SORTED PATHWAY' if args.use_sorted_pathway else 'CSV'}")
    if args.use_sorted_pathway:
        logger.info(f"  Sorted Pathway Dir: {args.sorted_pathway_dir}")
    elif args.csv_path:
        logger.info(f"  CSV: {args.csv_path}")
    logger.info(f"  Organs: {args.organs}")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Temperature: {args.temperature}, Seed: {args.seed}")
    logger.info(f"  Batch Size: {args.batch_size}, Max Retry: {args.max_retry}")
    logger.info(f"  Max Workers: {args.max_workers}")
    logger.info(f"  Ordering Mode: {args.ordering_mode}")
    logger.info(f"  Project: {args.project_name}")

    # Load data
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)

    df = None
    all_samples = None
    samples_by_source = None
    organs = args.organs

    if args.use_sorted_pathway:
        # Determine path based on task type
        if task_type == "task2" and args.case_study:
            sorted_base = os.path.join(
                args.sorted_pathway_dir, 
                args.model_name.replace("/", "_"),
                args.case_study
            )
        else:
            sorted_base = os.path.join(
                args.sorted_pathway_dir, 
                args.model_name.replace("/", "_")
            )
        
        logger.info(f"Loading from sorted pathway JSON files...")
        logger.info(f"  Base path: {sorted_base}")
        logger.info(f"  Organs: {organs}")
        
        all_samples, samples_by_source = load_sorted_pathway_data(
            sorted_pathway_dir=args.sorted_pathway_dir,
            model_name=args.model_name,
            task_type=task_type,
            seed=args.seed,
            organs=organs,
            case_study=args.case_study if task_type == "task2" else None,
        )
        logger.info(f"  Total samples: {len(all_samples)}")
        logger.info(f"  Source files: {len(samples_by_source)}")

        # Apply ordering mode (for ablation study)
        if args.ordering_mode != "default":
            logger.info(f"  Applying ordering mode: {args.ordering_mode}")
            if args.ordering_mode == "random":
                rng = random.Random(args.seed)
                for source_key in samples_by_source:
                    rng.shuffle(samples_by_source[source_key])
                # Rebuild all_samples in the new order
                all_samples = []
                for source_key in samples_by_source:
                    all_samples.extend(samples_by_source[source_key])
                logger.info(f"  Shuffled sample order within each source file (seed={args.seed})")
            elif args.ordering_mode == "reverse":
                for source_key in samples_by_source:
                    samples_by_source[source_key].reverse()
                # Rebuild all_samples in the new order
                all_samples = []
                for source_key in samples_by_source:
                    all_samples.extend(samples_by_source[source_key])
                logger.info(f"  Reversed sample order within each source file (hardest first)")

        # Create DataFrame for compatibility
        samples_list = [info.sample for info in all_samples]
        df = pd.DataFrame(samples_list)
    else:
        if not args.csv_path or not os.path.exists(args.csv_path):
            logger.error(f"CSV file not found: {args.csv_path}")
            sys.exit(1)
            
        df = load_benchmark_csv(args.csv_path, split=args.split)

        if args.cell_lines:
            logger.info(f"Filtering cell lines: {args.cell_lines}")
            df = df[df["cell_line"].isin(args.cell_lines)].copy()

        logger.info(f"  Total samples: {len(df)}")

    # Initialize components
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING COMPONENTS")
    logger.info("=" * 80)

    # Load domain knowledge
    global MOA_KNOWLEDGE, COMPOUND_KNOWLEDGE
    MOA_KNOWLEDGE, COMPOUND_KNOWLEDGE = load_domain_knowledge(
        "data/metadata/compound_moa/domain_knowledge_mapping.json"
    )
    logger.info(f"  Domain Knowledge: {len(MOA_KNOWLEDGE)} MoAs, {len(COMPOUND_KNOWLEDGE)} compounds")

    vllm_port = args.vllm_port or int(os.environ.get("VLLM_PORT", 8000))
    
    llm_client = VLLMClient(
        port=vllm_port,
        model=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
    )
    logger.info(f"  LLM client: port={vllm_port}, model={args.model_name}")

    unified_kg_context = UnifiedKGContext(max_pathways=args.max_pathways)
    use_depmap = args.use_depmap and not args.no_depmap

    knowledge_layer = KnowledgeRetrievalLayer(
        unified_kg_context=unified_kg_context,
        depmap_path=args.depmap_path,
        use_depmap=use_depmap,
        max_pathways=args.max_pathways,
    )
    logger.info(f"  DepMap context: {'enabled' if use_depmap else 'disabled'}")

    # Initialize history manager
    if args.use_history:
        logger.info(f"  Initializing STRING DB for history similarity...")
        stringdb = StringDBLocal()
        history_manager = HistoryManager(stringdb_loader=stringdb, topk=args.history_topk)
        logger.info(f"  History Manager: enabled (STRING DB similarity, topk={args.history_topk})")
    else:
        history_manager = HistoryManager(stringdb_loader=None, topk=args.history_topk)
        logger.info(f"  History Manager: enabled (tracking only, no STRING similarity)")

    # Set GAT checkpoint directory based on configuration
    gat_checkpoint_for_pipeline = args.gat_checkpoint_dir if use_gat else None

    pipeline = MultiScientistMultiJudgePipeline(
        llm_client=llm_client,
        knowledge_layer=knowledge_layer,
        max_pathways=args.max_pathways,
        max_retry=args.max_retry,
        history_manager=history_manager,
        task_type=task_type,
        gat_checkpoint_dir=gat_checkpoint_for_pipeline,
        device=args.gat_device,
        use_gat=use_gat,
        allow_uncertain=allow_uncertain,
    )
    logger.info(f"  Pipeline: initialized (max_retry={args.max_retry}, task_type={task_type})")
    logger.info(f"    Scientists: Context, Mechanism, Network")
    logger.info(f"    Judges: HistoryLeakage, TargetVerifier, Consistency, Logic")
    if gat_checkpoint_for_pipeline:
        logger.info(f"    GAT: enabled (checkpoint_dir={args.gat_checkpoint_dir})")
    else:
        logger.info(f"    GAT: disabled")
    if allow_uncertain:
        logger.info(f"    Answer Mode: yes/no/uncertain")
    else:
        logger.info(f"    Answer Mode: {'upregulated/downregulated' if task_type == 'task1' else 'yes/no'}")

    # Load GAT predictions
    logger.info("\n" + "=" * 80)
    logger.info("LOADING GAT PREDICTIONS")
    logger.info("=" * 80)

    if not use_gat:
        logger.info("GAT disabled: skipping GAT predictions")
        gat_predictions = {}
    elif args.use_sorted_pathway:
        # Sorted pathway mode: use runtime inference for both task1 and task2
        # No precomputed predictions available, pipeline will use checkpoint-based inference
        logger.info("Sorted pathway mode: Using runtime GAT inference from checkpoint")
        logger.info("  -> Pipeline will load GAT models per-organ and run inference on-the-fly")
        gat_predictions = {}
    elif args.use_precompute_gat:
        logger.info("Using PRECOMPUTED GAT predictions")
        gat_predictions = load_precomputed_gat_predictions(
            df=df,
            precompute_gat_dir=args.precompute_gat_dir,
            task_type=task_type,
            seed=args.seed,
        )
    else:
        logger.info("Loading GAT checkpoint for inference")
        train_csv = args.csv_path.replace("_test.csv", "_train.csv") if args.csv_path else None
        if train_csv and not os.path.exists(train_csv):
            train_csv = args.csv_path
        
        gat_predictions = load_all_gat_predictions(
            seed=args.seed,
            df=df,
            gat_checkpoint_dir=args.gat_checkpoint_dir,
            train_csv=train_csv,
            test_csv=args.csv_path,
            kg_dir=args.gat_kg_dir,
            kg_sources=args.gat_kg_sources,
            task_type=task_type,
            device=args.gat_device,
            organs=organs,
        ) if use_gat else {}

    logger.info(f"  Loaded {len(gat_predictions)} GAT predictions")

    # Run pipeline
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING MULTI-SCIENTIST MULTI-JUDGE PIPELINE")
    logger.info("=" * 80)

    if args.use_sorted_pathway:
        samples_by_organ: Dict[str, Dict[str, List[SortedSampleInfo]]] = {}
        for source_key, source_samples in samples_by_source.items():
            if not source_samples:
                continue
            organ = source_samples[0].organ
            samples_by_organ.setdefault(organ, {})[source_key] = source_samples

        results = run_sorted_pathway_pipeline(
            samples_by_organ=samples_by_organ,
            organs=organs,
            pipeline=pipeline,
            task_type=task_type,
            gat_predictions=gat_predictions,
            args=args,
        )
    else:
        results = run_csv_pipeline(
            df=df,
            pipeline=pipeline,
            task_type=task_type,
            gat_predictions=gat_predictions,
            args=args,
        )

    # Compute metrics
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)

    # Handle empty results case
    if not results:
        logger.warning("No results to evaluate. Check if data was loaded correctly.")
        logger.info(f"\nResults saved to: {args.results_dir}/{args.project_name}/")
        return
        
    # Complete message
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {args.results_dir}/{args.project_name}/")


if __name__ == "__main__":
    main()
