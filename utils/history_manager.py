"""
History Manager for STRING DB Similarity-based History.

Manages history of predictions for similar target genes based on STRING DB similarity.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HistoryManager:
    """
    Manages history of correct predictions for similar target genes.

    Tracks correct predictions per (cell_line, pert_gene) pair and retrieves
    top-k most similar target genes based on STRING DB similarity.
    """

    def __init__(self, stringdb_loader=None, topk: int = 3, min_similarity: float = 0.4):
        """
        Args:
            stringdb_loader: StringDBLocal instance for computing similarity
            topk: Number of similar history entries to return (default: 3)
            min_similarity: Minimum STRING DB similarity to include in history context (default: 0.4)
        """
        self.stringdb = stringdb_loader
        self.topk = topk
        self.min_similarity = min_similarity
        # History: {(cell_line, pert): [(target_gene, kg_context, predicted_result, is_correct, reasoning), ...]}
        self.history: Dict[Tuple, List[Dict]] = {}

    def _history_key(self, cell_line: str, pert: str, source_key: Optional[str] = None) -> Tuple:
        if source_key:
            return ("source", source_key)
        return (cell_line, pert)

    def add_result(
        self,
        cell_line: str,
        pert: str,
        target_gene: str,
        kg_context: str,
        predicted_result: str,
        is_correct: bool,
        source_key: Optional[str] = None,
        gt_label: Optional[int] = None,
        reasoning: Optional[str] = None,
    ):
        """Add ALL prediction results to history (regardless of correctness).

        Args:
            cell_line: Cell line name
            pert: Perturbation gene/compound
            target_gene: Target gene being predicted
            kg_context: Knowledge graph context used for this prediction
            predicted_result: Predicted answer (upregulated/downregulated or yes/no)
            is_correct: Whether prediction matched ground truth
            source_key: Optional source key for grouping
            gt_label: Ground truth label (0 or 1)
            reasoning: Integration agent's reasoning for this prediction
        """
        key = self._history_key(cell_line, pert, source_key=source_key)
        if key not in self.history:
            self.history[key] = []

        self.history[key].append({
            "target_gene": target_gene,
            "kg_context": kg_context,
            "predicted_result": predicted_result,
            "is_correct": is_correct,
            "gt_label": gt_label,
            "reasoning": reasoning,
        })
        logger.debug(f"History added: key={key}, gene={target_gene}, result={predicted_result}, total={len(self.history[key])}")

    def get_similar_history(
        self,
        cell_line: str,
        pert: str,
        target_gene: str,
        source_key: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get top-k most similar history entries based on STRING DB similarity.

        Args:
            cell_line: Current cell line
            pert: Current perturbation gene/compound
            target_gene: Current target gene

        Returns:
            List of history entries sorted by STRING similarity
        """
        key = self._history_key(cell_line, pert, source_key=source_key)
        if key not in self.history or not self.history[key]:
            return []

        history_entries = self.history[key]

        # If no STRING DB loader, return most recent entries
        if self.stringdb is None:
            return history_entries[-self.topk:]

        # Get all history target genes
        history_genes = [entry["target_gene"] for entry in history_entries]

        # Compute STRING similarity scores
        try:
            scores = self.stringdb.stringdb_score(target_gene, history_genes)
            gene_to_score = {gene: score for gene, score in scores}
        except Exception as e:
            logger.debug(f"STRING score computation failed: {e}")
            return history_entries[-self.topk:]

        # Sort history entries by STRING similarity (filtering by min_similarity)
        scored_entries = []
        for entry in history_entries:
            score = gene_to_score.get(entry["target_gene"], float("-inf"))
            if score >= self.min_similarity:
                scored_entries.append((score, entry))

        scored_entries.sort(key=lambda x: x[0], reverse=True)

        # Return top-k
        return [entry for _, entry in scored_entries[:self.topk]]

    def build_history_context(
        self,
        cell_line: str,
        pert: str,
        target_gene: str,
        source_key: Optional[str] = None,
        task_type: str = "task1",
    ) -> str:
        """
        Build history context string for integration agent prompt.

        Includes two sections:
        1. Summary of ALL (target_gene, prediction) pairs from same source
        2. Detailed context for TOP-K STRING-similar history genes including:
           - Predicted label
           - Knowledge graph context
           - Reasoning from integration agent (if available)

        For task2, converts yes/no predictions to upregulated/downregulated based on gt_label.
        """
        key = self._history_key(cell_line, pert, source_key=source_key)
        if key not in self.history or not self.history[key]:
            logger.debug(f"No history for key={key}, source_key={source_key}")
            return ""

        all_entries = self.history[key]
        logger.debug(f"Building history: key={key}, entries={len(all_entries)}")

        # Build gene-to-entry mapping and display predictions
        gene_to_entry = {}
        gene_display_preds = {}

        for entry in all_entries:
            gene = entry["target_gene"]
            pred = entry["predicted_result"]
            gt_label = entry.get("gt_label")

            # Store entry for later retrieval
            gene_to_entry[gene] = entry

            # For task2, convert yes/no to upregulated/downregulated
            if task_type == "task2" and gt_label is not None:
                if pred == "yes":
                    display_pred = "upregulated" if gt_label == 1 else "downregulated"
                elif pred == "no":
                    display_pred = "downregulated" if gt_label == 1 else "upregulated"
                else:
                    display_pred = pred
            else:
                display_pred = pred

            gene_display_preds[gene] = display_pred

        context_parts = []

        # Section 1: Summary of all predictions
        context_parts.append("== Other target genes' result ==")
        for gene, display_pred in gene_display_preds.items():
            context_parts.append(f"  - {gene}: {display_pred}")

        # Section 2: Detailed context for top-k STRING-similar genes
        if self.stringdb is not None:
            history_genes = list(gene_to_entry.keys())
            if history_genes:
                try:
                    scores = self.stringdb.stringdb_score(target_gene, history_genes)
                    if scores:
                        # Get top-k genes with similarity >= min_similarity
                        topk_genes = []
                        for gene, score in scores:
                            if score >= self.min_similarity and len(topk_genes) < self.topk:
                                topk_genes.append((gene, score))

                        if topk_genes:
                            context_parts.append(f"\n== Top-{len(topk_genes)} STRING-similar genes' detailed context ==")
                            context_parts.append(f"(These genes have high STRING DB similarity to {target_gene} and their reasoning may provide mechanistic insights)")

                            for idx, (sim_gene, sim_score) in enumerate(topk_genes, 1):
                                entry = gene_to_entry.get(sim_gene)
                                if entry:
                                    context_parts.append(f"\n--- Similar Gene {idx}: {sim_gene} (STRING similarity: {sim_score:.3f}) ---")
                                    context_parts.append(f"Prediction: {gene_display_preds.get(sim_gene, 'unknown')}")

                                    # Include KG context if available
                                    kg_ctx = entry.get("kg_context")
                                    if kg_ctx:
                                        # Truncate KG context if too long (keep first 2000 chars)
                                        if len(kg_ctx) > 2000:
                                            kg_ctx = kg_ctx[:2000] + "... [truncated]"
                                        context_parts.append(f"KG Context:\n{kg_ctx}")

                                    # Include reasoning if available
                                    reasoning = entry.get("reasoning")
                                    if reasoning:
                                        # Truncate reasoning if too long (keep first 1500 chars)
                                        if len(reasoning) > 1500:
                                            reasoning = reasoning[:1500] + "... [truncated]"
                                        context_parts.append(f"Reasoning:\n{reasoning}")
                except Exception as e:
                    logger.debug(f"STRING similarity lookup failed: {e}")

        return "\n".join(context_parts)

    def clear(self):
        """Clear all history."""
        self.history.clear()
