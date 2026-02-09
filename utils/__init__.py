"""
Utils module for LINCSQA Pipeline.

This module provides utilities for:
- LLM client (vLLM server interaction)
- Scientist and Judge agents
- Data loading (benchmark and sorted pathway)
- History management
- Result saving
- Configuration and utilities
"""

# LLM Client
from utils.llm_client import VLLMClient

# Agents
from utils.agents import (
    ContextAgent,
    MechanismAgent,
    NetworkAgent,
    IntegrationAgent,
)

# Judges
from utils.judges import (
    HistoryLeakageChecker,
    ReasoningTargetVerifier,
    ReasoningAnswerConsistencyChecker,
    ReasoningLogicChecker,
)

# Pipeline orchestrator
from utils.orchestrator import MultiScientistMultiJudgePipeline

# Data loaders
from utils.pipeline_data_loader import (
    SortedSampleInfo,
    load_sorted_pathway_data,
    CrossFileBatchIterator,
)
from utils.benchmark_data_loader import load_benchmark_csv

# History manager
from utils.history_manager import HistoryManager

# Configuration and utilities
from utils.config import (
    SampleResult,
    parse_llm_label_task2,
    build_gat_context,
    label_to_text,
    colorize_prediction,
    colorize_verdict,
    Colors,
    load_domain_knowledge,
    get_moa_knowledge,
    get_compound_knowledge,
)

# Result saver
from utils.result_saver import save_results_json, save_metrics_json

# Knowledge (mode3)
from utils.knowledge import (
    UnifiedKGContext,
    KnowledgeRetrievalLayer,
    filter_moa_from_depmap_context,
)

# GAT loader
from utils.gat_loader import (
    load_all_gat_predictions,
    load_precomputed_gat_predictions,
    GAT_AVAILABLE,
)

# Gene ordering metrics
from utils.gene_ordering_metrics import StringDBLocal

# Parsing
from utils.parsing import parse_llm_label

__all__ = [
    # LLM Client
    "VLLMClient",
    # Agents
    "ContextAgent",
    "MechanismAgent",
    "NetworkAgent",
    "IntegrationAgent",
    # Judges
    "HistoryLeakageChecker",
    "ReasoningTargetVerifier",
    "ReasoningAnswerConsistencyChecker",
    "ReasoningLogicChecker",
    # Pipeline
    "MultiScientistMultiJudgePipeline",
    # Data loaders
    "SortedSampleInfo",
    "load_sorted_pathway_data",
    "CrossFileBatchIterator",
    "load_benchmark_csv",
    # Metrics
    "compute_task1_auroc_per_organ",
    "compute_task2_topk_accuracy_per_organ",
    # History
    "HistoryManager",
    # Config
    "SampleResult",
    "parse_llm_label_task2",
    "build_gat_context",
    "label_to_text",
    "colorize_prediction",
    "colorize_verdict",
    "Colors",
    "load_domain_knowledge",
    "get_moa_knowledge",
    "get_compound_knowledge",
    # Result saver
    "save_results_json",
    "save_metrics_json",
    # Knowledge
    "UnifiedKGContext",
    "KnowledgeRetrievalLayer",
    "filter_moa_from_depmap_context",
    # GAT
    "load_all_gat_predictions",
    "load_precomputed_gat_predictions",
    "GAT_AVAILABLE",
    # Gene ordering
    "StringDBLocal",
    # Parsing
    "parse_llm_label",
]
