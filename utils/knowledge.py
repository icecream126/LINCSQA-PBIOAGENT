"""
Knowledge classes for Mode 3: GAT + LLM Ensemble.

Contains:
- HistoryEntry: Entry in progressive reasoning history buffer
- KnowledgeContext: External knowledge context for a sample
- UnifiedKGContext: Unified Knowledge Graph context generator
- SimilarityEngine: STRING DB-based similarity engine
- KnowledgeRetrievalLayer: External knowledge retrieval layer
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.kg_loader import KGLoader
from utils.gene_ordering_metrics import StringDBLocal

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class HistoryEntry:
    """Entry in the progressive reasoning history buffer."""

    gene: str
    reasoning: str
    answer: str
    go_terms: List[str] = field(default_factory=list)
    pathways: List[str] = field(default_factory=list)
    cell_line: str = ""


@dataclass
class KnowledgeContext:
    """External knowledge context for a sample."""

    moa: str
    targets: List[str]
    cell_line_info: str
    pathway_context: str = ""
    depmap_context: str = ""


# =============================================================================
# Constants
# =============================================================================

SIGNAL_ACTIVATION_MECHANISMS = {
    "phosphorylation",
    "dephosphorylation",
    "binding",
    "ubiquitination",
    "deubiquitination",
    "sumoylation",
    "acetylation",
    "deacetylation",
    "methylation",
    "demethylation",
    "cleavage",
    "gtpase-activating protein",
    "guanine nucleotide exchange factor",
    "catalytic activity",
}

TRANSCRIPTIONAL_CONTROL_MECHANISMS = {
    "transcriptional regulation",
    "transcriptional activation",
    "transcriptional repression",
    "post transcriptional regulation",
    "translation regulation",
    "mrna stabilization",
    "mrna destabilization",
}

CELL_LINE_TO_LINEAGE = {
    "MCF7": "Epithelial",
    "MDAMB231": "Epithelial",
    "SKBR3": "Epithelial",
    "HELA": "Epithelial",
    "A375": "Epithelial",
    "A549": "Epithelial",
    "HCC515": "Epithelial",
    "PC3": "Epithelial",
    "VCAP": "Epithelial",
    "HT29": "Epithelial",
    "HCT116": "Epithelial",
    "THP1": "Blood/Lymphoid",
    "K562": "Blood/Lymphoid",
    "JURKAT": "Blood/Lymphoid",
    "HL60": "Blood/Lymphoid",
    "NALM6": "Blood/Lymphoid",
    "OCILY19": "Blood/Lymphoid",
    "BJAB": "Blood/Lymphoid",
    "TMD8": "Blood/Lymphoid",
    "HBL1": "Blood/Lymphoid",
}


# =============================================================================
# Helper Functions
# =============================================================================


def classify_signor_mechanism(mechanism: str) -> str:
    """Classify SIGNOR mechanism into SIGNAL_ACTIVATION or TRANSCRIPTIONAL_CONTROL."""
    mechanism_lower = mechanism.lower().strip() if mechanism else ""
    if mechanism_lower in TRANSCRIPTIONAL_CONTROL_MECHANISMS:
        return "TRANSCRIPTIONAL_CONTROL"
    elif mechanism_lower in SIGNAL_ACTIVATION_MECHANISMS:
        return "SIGNAL_ACTIVATION"
    return "UNKNOWN"


def filter_moa_from_depmap_context(depmap_context: str) -> str:
    """Filter out MoA information from DepMap context."""
    if not depmap_context:
        return ""

    lines = depmap_context.split("\n")
    filtered_lines = []
    for line in lines:
        if re.search(r"Mechanism:\s*\w+", line, re.IGNORECASE):
            continue
        if re.search(r"known mechanism:", line, re.IGNORECASE):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


# =============================================================================
# Global StringDB Instance
# =============================================================================

_global_stringdb: Optional[StringDBLocal] = None


def get_global_stringdb() -> StringDBLocal:
    """Get or create global StringDBLocal instance (singleton pattern)."""
    global _global_stringdb
    if _global_stringdb is None:
        logger.info("Loading STRING DB (global instance, loaded once)...")
        _global_stringdb = StringDBLocal()
    return _global_stringdb


# =============================================================================
# Unified KG Context Generator
# =============================================================================


class UnifiedKGContext:
    """
    Unified Knowledge Graph context generator with mechanism-aware SIGNOR parsing.
    """

    def __init__(self, kg_loader: Optional[KGLoader] = None, max_pathways: int = 10):
        self._kg_loader = kg_loader
        self.max_pathways = max_pathways
        self._context_cache: Dict[str, str] = {}

    @property
    def kg_loader(self) -> KGLoader:
        """Lazy-load KGLoader."""
        if self._kg_loader is None:
            logger.info("Loading Knowledge Graph for unified context generation...")
            self._kg_loader = KGLoader(kg_dir="data/kg", use_uniprot=True)
        return self._kg_loader

    def generate_pathway_context(
        self,
        gene: str,
        max_pathways: Optional[int] = None,
        include_gene_info: bool = True,
    ) -> Tuple[str, List[str], List[str]]:
        """
        Generate unified KG pathway context for a gene.
        """
        max_pw = max_pathways or self.max_pathways
        pathway_statements = []
        go_terms = []
        pathways = []

        # 1. SIGNOR - Mechanism-aware regulatory relationships
        signor_graph = self.kg_loader.all_graphs.get("signor", {})
        if gene in signor_graph:
            for rel in signor_graph[gene][:4]:
                if isinstance(rel, (list, tuple)) and len(rel) >= 3:
                    target, effect, mechanism = (
                        rel[0],
                        rel[1],
                        rel[2] if len(rel) > 2 else "",
                    )
                    mech_class = classify_signor_mechanism(mechanism)
                    if mechanism:
                        pathway_statements.append(
                            f"{gene} {effect} {target} via {mechanism} [{mech_class}]"
                        )
                    else:
                        pathway_statements.append(f"{gene} {effect} {target}")
                    pathways.append(f"SIGNOR:{target}")

        # 2. GO relationships
        go_graph = self.kg_loader.all_graphs.get("go", {})
        if gene in go_graph:
            for rel in go_graph[gene][:4]:
                if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                    go_term, rel_type = rel[0], rel[1]
                    rel_readable = (
                        self.kg_loader.translate_go_rel(rel_type)
                        if hasattr(self.kg_loader, "translate_go_rel")
                        else rel_type
                    )
                    go_term_readable = (
                        self.kg_loader.translate_go_rel(go_term)
                        if hasattr(self.kg_loader, "translate_go_rel")
                        else go_term
                    )
                    pathway_statements.append(
                        f"{gene} {rel_readable} {go_term_readable}"
                    )
                    go_terms.append(go_term)

        # 3. Reactome pathways
        reactome_graph = self.kg_loader.all_graphs.get("reactome", {})
        if gene in reactome_graph:
            for rel in reactome_graph[gene][:3]:
                if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                    reaction_name, location = rel[0], rel[1]
                    pathway_statements.append(
                        f"{gene} participates in {reaction_name} in {location}"
                    )
                    pathways.append(reaction_name)

        # 4. STRING - Protein-protein interactions
        string_graph = self.kg_loader.all_graphs.get("string", {})
        if gene in string_graph:
            for rel in string_graph[gene][:3]:
                if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                    target_gene = rel[0]
                    evidence = rel[1]
                    if isinstance(evidence, (list, tuple)):
                        evidence_str = ", ".join(evidence[:2])
                    else:
                        evidence_str = str(evidence)
                    pathway_statements.append(
                        f"{gene} interacts with {target_gene} [evidence: {evidence_str}]"
                    )

        # 5. CORUM complexes
        corum_graph = self.kg_loader.all_graphs.get("corum", {})
        if gene in corum_graph:
            for rel in corum_graph[gene][:2]:
                if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                    complex_name, cell_type = rel[0], rel[1]
                    pathway_statements.append(
                        f"{gene} is member of {complex_name} in {cell_type}"
                    )
                    pathways.append(complex_name)

        # 6. BioPlex - Cell-type specific interactions
        bioplex_graph = self.kg_loader.all_graphs.get("bioplex", {})
        if gene in bioplex_graph:
            cell_interactions = {}
            for rel in bioplex_graph[gene][:6]:
                if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                    target_gene, cell_type = rel[0], rel[1]
                    if cell_type not in cell_interactions:
                        cell_interactions[cell_type] = []
                    cell_interactions[cell_type].append(target_gene)
            for cell_type, targets in list(cell_interactions.items())[:2]:
                targets_str = ", ".join(targets[:3])
                pathway_statements.append(
                    f"{gene} forms complex with {targets_str} in {cell_type}"
                )

        # Limit total pathways
        pathway_statements = pathway_statements[:max_pw]

        # Build context text
        context_parts = []

        if include_gene_info:
            # Ensembl - Gene description
            ensembl_desc = (
                self.kg_loader.ensembl_to_text(gene)
                if hasattr(self.kg_loader, "ensembl_to_text")
                else []
            )
            if ensembl_desc:
                context_parts.append(
                    f"Gene Info: {ensembl_desc[0].replace('Description of gene: ', '')}"
                )

            # UniProt - Protein functions
            uniprot_desc = (
                self.kg_loader.uniprot_to_text(gene)
                if hasattr(self.kg_loader, "uniprot_to_text")
                else []
            )
            if uniprot_desc:
                for desc in uniprot_desc[:2]:
                    if "Functions:" in desc:
                        context_parts.append(desc)
                        break

        # Add pathway context
        if pathway_statements:
            context_parts.append("Pathway context:")
            context_parts.extend(f"  {stmt}" for stmt in pathway_statements)
        else:
            context_parts.append("No pathway context available in knowledge graph.")

        context_text = "\n".join(context_parts)

        return context_text, go_terms, pathways


# =============================================================================
# Similarity Engine
# =============================================================================


class SimilarityEngine:
    """
    STRING DB-based similarity engine for history-based retrieval.
    """

    def __init__(
        self,
        kg_loader: Optional[KGLoader] = None,
        unified_kg_context: Optional[UnifiedKGContext] = None,
        stringdb: Optional[StringDBLocal] = None,
        max_pathways: int = 10,
        w_stringdb: float = 0.8,
        w_lineage: float = 0.2,
    ):
        self._kg_loader = kg_loader
        self._unified_kg_context = unified_kg_context
        self._stringdb = stringdb
        self.max_pathways = max_pathways
        self._stringdb_score_cache: Dict[Tuple[str, str], float] = {}
        self.w_stringdb = w_stringdb
        self.w_lineage = w_lineage
        self._current_cell_line: Optional[str] = None

    @property
    def kg_loader(self) -> KGLoader:
        """Lazy-load KGLoader."""
        if self._kg_loader is None:
            logger.info("Loading Knowledge Graph for similarity computation...")
            self._kg_loader = KGLoader(kg_dir="data/kg", use_uniprot=True)
        return self._kg_loader

    @property
    def unified_kg_context(self) -> UnifiedKGContext:
        """Lazy-load UnifiedKGContext."""
        if self._unified_kg_context is None:
            self._unified_kg_context = UnifiedKGContext(
                kg_loader=self.kg_loader, max_pathways=self.max_pathways
            )
        return self._unified_kg_context

    @property
    def stringdb(self) -> StringDBLocal:
        """Lazy-load StringDBLocal using global singleton instance."""
        if self._stringdb is None:
            self._stringdb = get_global_stringdb()
        return self._stringdb

    def set_cell_line(self, cell_line: str) -> None:
        """Set current cell line for lineage-aware similarity."""
        self._current_cell_line = cell_line

    def _compute_stringdb_score(self, gene1: str, gene2: str) -> float:
        """Compute STRING DB direct edge score between two genes."""
        cache_key = tuple(sorted([gene1, gene2]))
        if cache_key in self._stringdb_score_cache:
            return self._stringdb_score_cache[cache_key]

        results = self.stringdb.stringdb_score_direct(gene1, [gene2])

        score = 0.0
        for target, s in results:
            if target == gene2:
                score = s if s > 0 else 0.0
                break

        self._stringdb_score_cache[cache_key] = score
        return score

    def _compute_lineage_score(self, history_cell_line: Optional[str] = None) -> float:
        """Compute lineage match score: 1.0 if same lineage, else 0.0."""
        if not self._current_cell_line or not history_cell_line:
            return 0.0

        current_lineage = CELL_LINE_TO_LINEAGE.get(self._current_cell_line.upper())
        history_lineage = CELL_LINE_TO_LINEAGE.get(history_cell_line.upper())

        if current_lineage and history_lineage and current_lineage == history_lineage:
            return 1.0
        return 0.0

    def compute_similarity(
        self, gene1: str, gene2: str, history_cell_line: Optional[str] = None
    ) -> float:
        """Compute STRING DB-based weighted similarity."""
        stringdb_score = self._compute_stringdb_score(gene1, gene2)
        lineage = self._compute_lineage_score(history_cell_line)

        return self.w_stringdb * stringdb_score + self.w_lineage * lineage

    def get_top_k_similar(
        self, target_gene: str, history_buffer: List[HistoryEntry], k: int = 3
    ) -> List[Tuple[HistoryEntry, float]]:
        """Retrieve top-k most similar entries using STRING DB-based similarity."""
        if not history_buffer:
            return []

        similarities = []
        for entry in history_buffer:
            sim = self.compute_similarity(
                target_gene, entry.gene, history_cell_line=entry.cell_line
            )
            similarities.append((entry, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def generate_grounded_reasoning(
        self, gene: str, answer: str, max_pathways: int = None
    ) -> Tuple[str, List[str], List[str]]:
        """Generate KG-grounded pathway reasoning."""
        max_pw = max_pathways or self.max_pathways

        context_text, go_terms, pathways = (
            self.unified_kg_context.generate_pathway_context(
                gene=gene, max_pathways=max_pw, include_gene_info=True
            )
        )

        reasoning_text = f"Prediction: {answer}\n\n{context_text}"

        return reasoning_text, go_terms, pathways


# =============================================================================
# Knowledge Retrieval Layer
# =============================================================================


class KnowledgeRetrievalLayer:
    """
    Retrieve external domain knowledge for knowledge enhancement.
    """

    CELL_LINE_METADATA_PATH = "data/metadata/cell/cellosaurus.json"
    DEPMAP_DATA_PATH = "data/kg/depmap"

    def __init__(
        self,
        kg_loader: Optional[KGLoader] = None,
        unified_kg_context: Optional[UnifiedKGContext] = None,
        cell_line_metadata_path: Optional[str] = None,
        depmap_path: Optional[str] = None,
        use_depmap: bool = True,
        max_pathways: int = 10,
    ):
        self._kg_loader = kg_loader
        self._unified_kg_context = unified_kg_context
        self.max_pathways = max_pathways
        self._cell_line_info: Dict[str, str] = {}
        self._use_depmap = use_depmap
        self._depmap_generator = None

        metadata_path = cell_line_metadata_path or self.CELL_LINE_METADATA_PATH
        self._load_cell_line_metadata(metadata_path)

        if use_depmap:
            self._init_depmap(depmap_path)

    def _init_depmap(self, depmap_path: Optional[str] = None) -> None:
        """Initialize DepMap context generator."""
        try:
            from scripts.depmap import ContextBlockGenerator
            from scripts.depmap.context_generator import ContextBlockConfig

            depmap_data_path = depmap_path or self.DEPMAP_DATA_PATH
            config = ContextBlockConfig(
                include_essentiality=False,
                include_compensation_bias=True,
                include_target_importance=True,
            )
            self._depmap_generator = ContextBlockGenerator(depmap_data_path, config=config)
            logger.info(
                f"Initialized DepMap context generator from {depmap_data_path} (essentiality excluded)"
            )
        except ImportError:
            logger.warning("DepMap module not available")
            self._depmap_generator = None
        except Exception as e:
            logger.warning(f"Failed to initialize DepMap context generator: {e}")
            self._depmap_generator = None

    def _load_cell_line_metadata(self, metadata_path: str) -> None:
        """Load cell line metadata from JSON file."""
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            for cell_line_key, info_str in raw_data.items():
                self._cell_line_info[cell_line_key.lower()] = info_str
                self._cell_line_info[cell_line_key.upper()] = info_str

            logger.info(
                f"Loaded {len(raw_data)} cell line entries from {metadata_path}"
            )

        except FileNotFoundError:
            logger.warning(f"Cell line metadata file not found: {metadata_path}")
        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing cell line metadata JSON: {e}")

    def _get_cell_line_info(self, cell_line: str) -> str:
        """Get cell line information from loaded metadata."""
        cell_line_key = cell_line.lower()
        if cell_line_key in self._cell_line_info:
            return self._cell_line_info[cell_line_key]

        cell_line_key = cell_line.upper()
        if cell_line_key in self._cell_line_info:
            return self._cell_line_info[cell_line_key]

        return f"{cell_line} is a cell line used in biological research."

    @property
    def kg_loader(self) -> KGLoader:
        """Lazy-load KGLoader."""
        if self._kg_loader is None:
            logger.info("Loading Knowledge Graph for knowledge retrieval...")
            self._kg_loader = KGLoader(kg_dir="data/kg", use_uniprot=True)
        return self._kg_loader

    @property
    def unified_kg_context(self) -> UnifiedKGContext:
        """Lazy-load UnifiedKGContext."""
        if self._unified_kg_context is None:
            self._unified_kg_context = UnifiedKGContext(
                kg_loader=self.kg_loader, max_pathways=self.max_pathways
            )
        return self._unified_kg_context

    def retrieve_knowledge(
        self,
        pert: str,
        gene: str,
        cell_line: str,
        gt_moa: Optional[str] = None,
        candidate_moa: Optional[str] = None,
        compound: Optional[str] = None,
        max_pathways: Optional[int] = None,
    ) -> KnowledgeContext:
        """Retrieve external knowledge for a sample."""
        moa = gt_moa if gt_moa and gt_moa != "unknown" else candidate_moa
        if not moa or moa == "unknown":
            moa = "Unknown mechanism of action"

        pathway_context, go_terms, pathways = (
            self.unified_kg_context.generate_pathway_context(
                gene=gene,
                max_pathways=max_pathways or self.max_pathways,
                include_gene_info=True,
            )
        )

        targets = []
        string_graph = self.kg_loader.all_graphs.get("string", {})
        if gene in string_graph:
            rels = string_graph[gene]
            for rel in rels[:10]:
                if isinstance(rel, (list, tuple)) and len(rel) >= 1:
                    targets.append(rel[0])

        cell_line_info = self._get_cell_line_info(cell_line)

        depmap_context = ""
        if self._use_depmap and self._depmap_generator is not None:
            try:
                compound_name = compound or self._extract_compound_from_pert(pert)
                depmap_context = self._depmap_generator.generate_for_regulation_query(
                    cell_line=cell_line, compound=compound_name, query_gene=gene
                )
            except Exception as e:
                logger.debug(f"Failed to get DepMap context: {e}")
                depmap_context = ""

        return KnowledgeContext(
            moa=moa,
            targets=targets,
            cell_line_info=cell_line_info,
            pathway_context=pathway_context,
            depmap_context=depmap_context,
        )

    def _extract_compound_from_pert(self, pert: str) -> str:
        """Extract compound name from perturbation string."""
        if "(" in pert and ")" in pert:
            start = pert.rfind("(") + 1
            end = pert.rfind(")")
            if start < end:
                return pert[start:end]
        return pert
