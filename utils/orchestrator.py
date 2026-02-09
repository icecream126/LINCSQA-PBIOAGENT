"""
Pipeline Orchestrator for Multi-Scientist Multi-Judge Pipeline.

This module contains the main pipeline class that coordinates:
1. 3 Scientist Agents (Context, Mechanism, Network) - run in PARALLEL
2. Integration Agent - receives all scientist reasoning + GAT prediction
3. 4 Judge Agents (HistoryLeakage, TargetVerifier, Consistency, Logic) - run in PARALLEL
4. Retry Logic - retry if any judge returns "problematic", up to max_retry times
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from utils.parsing import parse_llm_label
from utils.config import (
    SampleResult,
    parse_llm_label_task2,
    label_to_text,
    colorize_prediction,
    colorize_verdict,
)
from utils.agents import (
    ContextAgent,
    MechanismAgent,
    NetworkAgent,
    IntegrationAgent,
)
from utils.judges import (
    HistoryLeakageChecker,
    ReasoningTargetVerifier,
    ReasoningAnswerConsistencyChecker,
    ReasoningLogicChecker,
)
from utils.history_manager import HistoryManager

logger = logging.getLogger(__name__)


class MultiScientistMultiJudgePipeline:
    """
    Complete pipeline with scientists, integration, and judges.
    
    For each sample with retry:
    1. Run 3 scientists in PARALLEL
    2. Run Integration Agent
    3. Run 4 Judges in PARALLEL
    4. If all pass -> done; else retry up to max_retry
    """

    def __init__(
        self,
        llm_client,
        knowledge_layer,
        max_pathways: int = 10,
        max_retry: int = 2,
        history_manager: Optional[HistoryManager] = None,
        task_type: str = "task1",
        gat_checkpoint_dir: Optional[str] = None,
        device: str = "cuda",
        use_gat: bool = True,
        allow_uncertain: bool = False,
    ):
        self.llm_client = llm_client
        self.knowledge_layer = knowledge_layer
        self.max_pathways = max_pathways
        self.max_retry = max_retry
        self.history_manager = history_manager
        self.task_type = task_type
        self.gat_checkpoint_dir = gat_checkpoint_dir
        self.device = device
        self.use_gat = use_gat
        self.allow_uncertain = allow_uncertain

        # Initialize scientist agents
        self.context_agent = ContextAgent(llm_client)
        self.mechanism_agent = MechanismAgent(llm_client)
        self.network_agent = NetworkAgent(llm_client)
        self.integration_agent = IntegrationAgent(llm_client)

        # Initialize judge agents
        self.history_leakage_checker = HistoryLeakageChecker(llm_client)
        self.target_verifier = ReasoningTargetVerifier(llm_client)
        self.consistency_checker = ReasoningAnswerConsistencyChecker(llm_client)
        self.logic_checker = ReasoningLogicChecker(llm_client)
        
        # Cache for loaded GAT models (per organ)
        self._gat_models: Dict[str, Any] = {}
        self._gat_tokenizers: Dict[str, Dict] = {}

    def _get_kg_context(self, gene: str) -> str:
        """Get KG context for a gene (includes gene info and pathway context)."""
        try:
            pathway_context, _, _ = self.knowledge_layer.unified_kg_context.generate_pathway_context(
                gene=gene,
                max_pathways=self.max_pathways,
                include_gene_info=True,
            )
            return pathway_context if pathway_context else ""
        except Exception as e:
            logger.debug(f"Failed to get KG context: {e}")
            return ""

    def _get_cell_line_info(self, cell_line: str) -> str:
        """Get cell line information from knowledge layer."""
        try:
            return self.knowledge_layer._get_cell_line_info(cell_line)
        except Exception as e:
            logger.debug(f"Failed to get cell line info: {e}")
            return f"{cell_line} is a cell line used in biological research."

    def _get_depmap_context(self, pert: str, gene: str, cell_line: str, 
                            gt_moa: str, candidate_moa: str, compound: str) -> str:
        """Get DepMap context for a sample."""
        try:
            knowledge = self.knowledge_layer.retrieve_knowledge(
                pert=pert, gene=gene, cell_line=cell_line,
                gt_moa=gt_moa, candidate_moa=candidate_moa, compound=compound,
            )
            return knowledge.depmap_context
        except Exception:
            return ""

    def _load_gat_model_for_organ(self, organ: str) -> bool:
        """
        Load GAT model for a specific organ from checkpoint.
        
        Returns True if model was loaded successfully, False otherwise.
        """
        if organ in self._gat_models:
            return True
        
        if not self.gat_checkpoint_dir:
            return False
        
        try:
            import torch
            import torch.nn.functional as F
            from baselines.run_gat import GAT as GATModel
            from torch_geometric.data import Data
            
            device_obj = torch.device(self.device if torch.cuda.is_available() else "cpu")
            
            # Always load task1 checkpoint (GAT models are trained only on task1)
            checkpoint_path = os.path.join(
                self.gat_checkpoint_dir, "task1", organ, "best_model_0.pt"
            )
            
            if not os.path.exists(checkpoint_path):
                logger.warning(f"GAT checkpoint not found: {checkpoint_path}")
                return False
            
            checkpoint = torch.load(checkpoint_path, map_location=device_obj)
            saved_args = checkpoint.get("args", {})
            
            # Get tokenizers from checkpoint
            saved_tokenizer = checkpoint.get("tokenizer", {})
            saved_cell_line_tokenizer = checkpoint.get("cell_line_tokenizer", {})
            saved_moa_tokenizer = checkpoint.get("moa_tokenizer", {})
            saved_compound_tokenizer = checkpoint.get("compound_tokenizer", {})
            
            if not saved_tokenizer:
                logger.warning(f"Checkpoint for organ {organ} does not contain tokenizer")
                return False
            
            # Store tokenizers
            self._gat_tokenizers[organ] = {
                "tokenizer": saved_tokenizer,
                "cell_line_tokenizer": saved_cell_line_tokenizer,
                "moa_tokenizer": saved_moa_tokenizer,
                "compound_tokenizer": saved_compound_tokenizer,
            }
            
            # Reconstruct model
            state_dict = checkpoint["model_state_dict"]
            checkpoint_edge_index = state_dict["edge_index"]
            if checkpoint_edge_index.shape[0] == 2:
                checkpoint_edge_index = checkpoint_edge_index.T
            
            checkpoint_graph = Data(
                x=state_dict["x"],
                edge_index=checkpoint_edge_index,
                edge_attr=state_dict["edge_attr"],
            )
            
            embed_dim = saved_args.get("embed_dim", 256)
            ffn_embed_dim = saved_args.get("ffn_embed_dim", 1024)
            num_layers = saved_args.get("num_layers", 4)
            dropout = saved_args.get("dropout", 0.1)
            vocab_size = saved_args.get("vocab_size") or len(saved_tokenizer)
            num_kgs = saved_args.get("num_kgs") or 1
            num_cell_lines = saved_args.get("num_cell_lines") or len(saved_cell_line_tokenizer)
            num_moas = saved_args.get("num_moas") or len(saved_moa_tokenizer) or 1
            num_compounds = saved_args.get("num_compounds") or len(saved_compound_tokenizer) or 1
            
            model = GATModel(
                vocab_size=vocab_size,
                num_kgs=num_kgs,
                embed_dim=embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                num_layers=num_layers,
                dropout=dropout,
                graph=checkpoint_graph,
                num_cell_lines=num_cell_lines,
                num_moas=num_moas,
                num_compounds=num_compounds,
            ).to(device_obj)
            
            model.load_state_dict(state_dict)
            model.eval()
            
            self._gat_models[organ] = {
                "model": model,
                "device": device_obj,
            }
            
            logger.info(f"  Loaded GAT model for organ {organ} from checkpoint")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load GAT model for organ {organ}: {e}")
            return False

    def _infer_gat_single_sample(self, sample: Dict) -> Optional[float]:
        """
        Run GAT inference for a single sample using loaded checkpoint.
        
        Returns probability or None if inference fails.
        """
        organ = sample.get("organ", "").lower().replace(" ", "_")
        if not organ:
            return None
        
        if not self._load_gat_model_for_organ(organ):
            return None
        
        try:
            import torch
            import torch.nn.functional as F
            
            model_info = self._gat_models[organ]
            model = model_info["model"]
            device_obj = model_info["device"]
            tokenizers = self._gat_tokenizers[organ]
            
            # Extract perturbation entity
            pert_raw = sample.get("pert", "")
            if "(" in pert_raw and ")" in pert_raw:
                pert_entity = pert_raw[pert_raw.index("(") + 1:pert_raw.index(")")]
            else:
                pert_entity = pert_raw
            
            gene_raw = sample.get("gene", "")
            cell_line = sample.get("cell_line", "")
            candidate_moa = sample.get("candidate_moa", "")
            compound = sample.get("compound", pert_entity)
            
            # Tokenize
            pert_idx = tokenizers["tokenizer"].get(pert_entity, 0)
            gene_idx = tokenizers["tokenizer"].get(gene_raw, 0)
            cell_line_idx = tokenizers["cell_line_tokenizer"].get(cell_line, 0)
            compound_idx = tokenizers["compound_tokenizer"].get(compound, 0)
            moa_idx = tokenizers["moa_tokenizer"].get(candidate_moa, 0)
            
            # Create batch
            batch = {
                "pert": torch.tensor([pert_idx], dtype=torch.long).to(device_obj),
                "gene": torch.tensor([gene_idx], dtype=torch.long).to(device_obj),
                "cell_line_idx": torch.tensor([cell_line_idx], dtype=torch.long).to(device_obj),
                "moa": torch.tensor([moa_idx], dtype=torch.long).to(device_obj),
                "compound": torch.tensor([compound_idx], dtype=torch.long).to(device_obj),
            }
            
            with torch.no_grad():
                output = model(batch)
                prob = F.softmax(output, dim=-1)[0, 1].cpu().item()
            
            return float(prob)
            
        except Exception as e:
            logger.debug(f"GAT inference failed for sample: {e}")
            return None

    def process_sample(
        self,
        sample: Dict,
        gat_pred: Optional[Dict] = None,
        max_workers: int = 4,
    ) -> SampleResult:
        """
        Process a single sample through the complete pipeline.
        
        Args:
            sample: Sample dictionary with pert, gene, cell_line, label, etc.
            gat_pred: Optional precomputed GAT prediction
            max_workers: Number of parallel workers for scientist/judge agents
            
        Returns:
            SampleResult with prediction and all intermediate outputs
        """
        import time
        start_time = time.time()
        
        # Extract sample info
        pert = sample.get("pert", "")
        gene = sample.get("gene", "")
        cell_line = sample.get("cell_line", "")
        label = sample.get("label", 0)
        gt_moa = sample.get("gt_moa", "")
        candidate_moa = sample.get("candidate_moa", "")
        compound = sample.get("compound", "")
        source_key = sample.get("source_key", "")
        
        # Get contexts
        kg_context = self._get_kg_context(gene)
        cell_line_info = self._get_cell_line_info(cell_line)
        depmap_context = self._get_depmap_context(pert, gene, cell_line, gt_moa, candidate_moa, compound)
        
        # Get GAT prediction
        gat_prob = None
        if self.use_gat:
            if gat_pred is not None:
                gat_prob = gat_pred.get("confidence", 0.5)
            else:
                gat_prob = self._infer_gat_single_sample(sample)
        
        gat_context = self._build_gat_context(gat_prob, gene)
        
        # Get history context
        history_context = ""
        if self.history_manager:
            history_context = self.history_manager.build_history_context(
                cell_line=cell_line,
                pert=pert,
                target_gene=gene,
                source_key=source_key,
                task_type=self.task_type,
            )
        
        # Initialize result tracking
        scientists_reasoning = {}
        judges_verdict = {}
        agents_input_prompts = {}
        judges_input_prompts = {}
        final_answer = None
        final_reasoning = ""
        retry_count = 0

        # Retry loop
        judge_feedback = ""
        while retry_count <= self.max_retry:
            # Run scientists in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._run_context_agent_with_prompt, cell_line, pert, gene,
                                   cell_line_info, kg_context, depmap_context, compound): "context",
                    executor.submit(self._run_mechanism_agent_with_prompt, pert, gene, compound,
                                   cell_line_info, kg_context): "mechanism",
                    executor.submit(self._run_network_agent_with_prompt, pert, gene, cell_line_info,
                                   kg_context, compound): "network",
                }

                for future in as_completed(futures):
                    agent_name = futures[future]
                    try:
                        reasoning, input_prompt = future.result()
                        scientists_reasoning[agent_name] = reasoning
                        agents_input_prompts[agent_name] = input_prompt
                    except Exception as e:
                        logger.error(f"Scientist {agent_name} failed: {e}")
                        scientists_reasoning[agent_name] = ""
                        agents_input_prompts[agent_name] = ""
            
            # Run integration agent
            final_reasoning, final_answer, integration_input_prompt = self._run_integration_agent_with_prompt(
                cell_line=cell_line,
                pert=pert,
                target_gene=gene,
                context_reasoning=scientists_reasoning.get("context", ""),
                mechanism_reasoning=scientists_reasoning.get("mechanism", ""),
                network_reasoning=scientists_reasoning.get("network", ""),
                gat_context=gat_context,
                history_context=history_context,
                cell_line_info=cell_line_info,
                kg_context=kg_context,
                judge_feedback=judge_feedback,
                compound=compound,
                candidate_moa=candidate_moa,
                label=label,
            )
            agents_input_prompts["integration"] = integration_input_prompt

            # Run judges in parallel
            all_passed = True
            judge_feedback_parts = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                judge_futures = {
                    executor.submit(self._run_history_leakage_checker_with_prompt,
                                   history_context, final_reasoning, final_answer): "history_leakage",
                    executor.submit(self._run_target_verifier_with_prompt,
                                   cell_line, pert, gene, final_reasoning, final_answer): "target_verifier",
                    executor.submit(self._run_consistency_checker_with_prompt,
                                   final_reasoning, final_answer, gene, label): "consistency",
                    executor.submit(self._run_logic_checker_with_prompt,
                                   cell_line, pert, gene, final_reasoning, final_answer): "logic",
                }

                for future in as_completed(judge_futures):
                    judge_name = judge_futures[future]
                    try:
                        verdict, judge_input_prompt = future.result()
                        judges_verdict[judge_name] = verdict
                        judges_input_prompts[judge_name] = judge_input_prompt
                        if verdict.get("verdict") == "problematic":
                            all_passed = False
                            if verdict.get("feedback"):
                                judge_feedback_parts.append(f"[{judge_name}] {verdict['feedback']}")
                    except Exception as e:
                        logger.error(f"Judge {judge_name} failed: {e}")
                        judges_verdict[judge_name] = {"verdict": "error", "feedback": str(e)}
                        judges_input_prompts[judge_name] = ""
            
            if all_passed:
                break
            
            retry_count += 1
            judge_feedback = "\n".join(judge_feedback_parts)
        
        # Determine correctness
        if self.task_type == "task1":
            is_correct = (final_answer == "upregulated" and label == 1) or \
                        (final_answer == "downregulated" and label == 0)
            predicted_label = 1 if final_answer == "upregulated" else 0
        else:
            is_correct = (final_answer == "yes" and label == 1) or \
                        (final_answer == "no" and label == 0)
            predicted_label = final_answer  # Keep as string for task2
        
        # Add to history (including reasoning for future reference)
        if self.history_manager:
            self.history_manager.add_result(
                cell_line=cell_line,
                pert=pert,
                target_gene=gene,
                kg_context=kg_context,
                predicted_result=final_answer,
                is_correct=is_correct,
                source_key=source_key,
                gt_label=label,
                reasoning=final_reasoning,
            )
        
        processing_time = time.time() - start_time

        # Determine gat_label from gat_prob
        gat_label = None
        if gat_prob is not None:
            gat_label = 1 if gat_prob > 0.5 else 0

        return SampleResult(
            sample=sample,
            predicted_label=predicted_label,
            ground_truth=label,
            reasoning=final_reasoning,
            scientists_reasoning=scientists_reasoning,
            judges_verdict=judges_verdict,
            retry_count=retry_count,
            is_correct=is_correct,
            processing_time=processing_time,
            gat_prob=gat_prob,
            gat_label=gat_label,
            answer=final_answer,
            agents_input_prompts=agents_input_prompts,
            judges_input_prompts=judges_input_prompts,
        )

    def process_batch(
        self,
        samples: List[Dict],
        gat_predictions: Optional[Dict[str, Dict]] = None,
        parallelize: bool = True,
    ) -> List[SampleResult]:
        """
        Process a batch of samples.
        
        Args:
            samples: List of sample dictionaries
            gat_predictions: Optional dict mapping sample keys to GAT predictions
            parallelize: Whether to process samples in parallel
            
        Returns:
            List of SampleResult objects
        """
        gat_predictions = gat_predictions or {}
        
        def _process_single(sample: Dict) -> SampleResult:
            """Process a single sample with GAT prediction lookup."""
            # Build sample key for GAT prediction lookup
            gene = sample.get("gene", "")
            cell_line = sample.get("cell_line", "")
            pert = sample.get("pert", "")
            sample_key = f"{gene}_{cell_line}_{pert}"
            
            gat_pred = gat_predictions.get(sample_key)
            return self.process_sample(sample, gat_pred=gat_pred)
        
        if parallelize and len(samples) > 1:
            # Process in parallel using llm_client's max_workers
            max_workers = min(len(samples), self.llm_client.max_workers)
            results = [None] * len(samples)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_process_single, sample): idx
                    for idx, sample in enumerate(samples)
                }
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Sample {idx} processing failed: {e}")
                        # Create error result
                        sample = samples[idx]
                        results[idx] = SampleResult(
                            sample=sample,
                            predicted_label=0 if self.task_type == "task1" else "no",
                            ground_truth=sample.get("label", 0),
                            reasoning=f"Error: {e}",
                            scientists_reasoning={},
                            judges_verdict={},
                            retry_count=0,
                            is_correct=False,
                            processing_time=0.0,
                        )
            
            return results
        else:
            # Process sequentially
            return [_process_single(sample) for sample in samples]

    def _build_gat_context(self, gat_prob: Optional[float], gene: str) -> str:
        """Build GAT prediction context string."""
        if gat_prob is None:
            return "GAT prediction: Not available"
        
        if self.task_type == "task1":
            pred_label = "upregulated" if gat_prob > 0.5 else "downregulated"
        else:
            pred_label = "yes" if gat_prob > 0.5 else "no"
        
        return f"GAT prediction for {gene}: {pred_label} (confidence: {gat_prob:.2%})"

    def _run_context_agent(self, cell_line, pert, gene, cell_line_info, kg_context, depmap_context, compound):
        """Run context agent and return reasoning."""
        system_prompt, user_prompt = self.context_agent.build_prompt(
            cell_line=cell_line, pert=pert, target_gene=gene,
            cell_line_info=cell_line_info, kg_context=kg_context,
            depmap_context=depmap_context, compound=compound,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.llm_client.generate(messages)

    def _run_mechanism_agent(self, pert, gene, compound, cell_line_info, kg_context):
        """Run mechanism agent and return reasoning."""
        system_prompt, user_prompt = self.mechanism_agent.build_prompt(
            pert=pert, target_gene=gene, compound=compound,
            cell_line_info=cell_line_info, kg_context=kg_context,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.llm_client.generate(messages)

    def _run_network_agent(self, pert, gene, cell_line_info, kg_context, compound):
        """Run network agent and return reasoning."""
        system_prompt, user_prompt = self.network_agent.build_prompt(
            pert=pert, target_gene=gene, cell_line_info=cell_line_info,
            kg_context=kg_context, compound=compound,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.llm_client.generate(messages)

    def _run_integration_agent(self, cell_line, pert, target_gene, context_reasoning,
                               mechanism_reasoning, network_reasoning, gat_context,
                               history_context, cell_line_info, kg_context, judge_feedback,
                               compound, candidate_moa, label):
        """Run integration agent and return (reasoning, answer)."""
        system_prompt, user_prompt = self.integration_agent.build_prompt(
            cell_line=cell_line, pert=pert, target_gene=target_gene,
            context_reasoning=context_reasoning, mechanism_reasoning=mechanism_reasoning,
            network_reasoning=network_reasoning, gat_context=gat_context,
            history_context=history_context, cell_line_info=cell_line_info,
            kg_context=kg_context, judge_feedback=judge_feedback,
            compound=compound, candidate_moa=candidate_moa,
            task_type=self.task_type, label=label,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.generate(messages)
        
        # Parse answer
        if self.task_type == "task1":
            answer = parse_llm_label(response)
        else:
            answer = parse_llm_label_task2(response, allow_uncertain=self.allow_uncertain)
        
        return response, answer

    def _run_history_leakage_checker(self, history_context, final_reasoning, final_answer):
        """Run history leakage checker."""
        system_prompt, user_prompt = self.history_leakage_checker.build_prompt(
            history_summary=history_context,
            reasoning=final_reasoning,
            answer=final_answer,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.generate(messages)
        return self.history_leakage_checker.parse_response(response)

    def _run_target_verifier(self, cell_line, pert, gene, final_reasoning, final_answer):
        """Run target verifier."""
        system_prompt, user_prompt = self.target_verifier.build_prompt(
            cell_line=cell_line, pert=pert, target_gene=gene,
            reasoning=final_reasoning,
            answer=final_answer,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.generate(messages)
        return self.target_verifier.parse_response(response)

    def _run_consistency_checker(self, final_reasoning, final_answer, gene, label):
        """Run consistency checker."""
        if self.task_type == "task2":
            direction = "upregulated" if label == 1 else "downregulated"
            question = f"Will {gene} be {direction}?"
        else:
            question = ""
            direction = ""

        system_prompt, user_prompt = self.consistency_checker.build_prompt(
            reasoning=final_reasoning,
            answer=final_answer,
            task_type=self.task_type,
            question=question,
            ground_truth_direction=direction,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.generate(messages)
        return self.consistency_checker.parse_response(response)

    def _run_logic_checker(self, cell_line, pert, gene, final_reasoning, final_answer):
        """Run logic checker."""
        if self.task_type == "task1":
            question = f"In {cell_line}, will {gene} be upregulated or downregulated by {pert}?"
        else:
            question = f"Will {gene} be affected by {pert}?"

        system_prompt, user_prompt = self.logic_checker.build_prompt(
            question=question,
            answer=final_answer,
            reasoning=final_reasoning,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.generate(messages)
        return self.logic_checker.parse_response(response)

    # ==================== Methods with prompt capture ====================

    def _format_input_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format system and user prompts into a single string for logging."""
        return f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"

    def _run_context_agent_with_prompt(self, cell_line, pert, gene, cell_line_info, kg_context, depmap_context, compound):
        """Run context agent and return (reasoning, input_prompt)."""
        system_prompt, user_prompt = self.context_agent.build_prompt(
            cell_line=cell_line, pert=pert, target_gene=gene,
            cell_line_info=cell_line_info, kg_context=kg_context,
            depmap_context=depmap_context, compound=compound,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.generate(messages)
        input_prompt = self._format_input_prompt(system_prompt, user_prompt)
        return response, input_prompt

    def _run_mechanism_agent_with_prompt(self, pert, gene, compound, cell_line_info, kg_context):
        """Run mechanism agent and return (reasoning, input_prompt)."""
        system_prompt, user_prompt = self.mechanism_agent.build_prompt(
            pert=pert, target_gene=gene, compound=compound,
            cell_line_info=cell_line_info, kg_context=kg_context,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.generate(messages)
        input_prompt = self._format_input_prompt(system_prompt, user_prompt)
        return response, input_prompt

    def _run_network_agent_with_prompt(self, pert, gene, cell_line_info, kg_context, compound):
        """Run network agent and return (reasoning, input_prompt)."""
        system_prompt, user_prompt = self.network_agent.build_prompt(
            pert=pert, target_gene=gene, cell_line_info=cell_line_info,
            kg_context=kg_context, compound=compound,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.generate(messages)
        input_prompt = self._format_input_prompt(system_prompt, user_prompt)
        return response, input_prompt

    def _run_integration_agent_with_prompt(self, cell_line, pert, target_gene, context_reasoning,
                                           mechanism_reasoning, network_reasoning, gat_context,
                                           history_context, cell_line_info, kg_context, judge_feedback,
                                           compound, candidate_moa, label):
        """Run integration agent and return (reasoning, answer, input_prompt)."""
        system_prompt, user_prompt = self.integration_agent.build_prompt(
            cell_line=cell_line, pert=pert, target_gene=target_gene,
            context_reasoning=context_reasoning, mechanism_reasoning=mechanism_reasoning,
            network_reasoning=network_reasoning, gat_context=gat_context,
            history_context=history_context, cell_line_info=cell_line_info,
            kg_context=kg_context, judge_feedback=judge_feedback,
            compound=compound, candidate_moa=candidate_moa,
            task_type=self.task_type, label=label,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.generate(messages)
        input_prompt = self._format_input_prompt(system_prompt, user_prompt)

        # Parse answer
        if self.task_type == "task1":
            answer = parse_llm_label(response)
        else:
            answer = parse_llm_label_task2(response, allow_uncertain=self.allow_uncertain)

        return response, answer, input_prompt

    def _run_history_leakage_checker_with_prompt(self, history_context, final_reasoning, final_answer):
        """Run history leakage checker and return (verdict, input_prompt)."""
        system_prompt, user_prompt = self.history_leakage_checker.build_prompt(
            history_summary=history_context,
            reasoning=final_reasoning,
            answer=final_answer,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.generate(messages)
        input_prompt = self._format_input_prompt(system_prompt, user_prompt)
        return self.history_leakage_checker.parse_response(response), input_prompt

    def _run_target_verifier_with_prompt(self, cell_line, pert, gene, final_reasoning, final_answer):
        """Run target verifier and return (verdict, input_prompt)."""
        system_prompt, user_prompt = self.target_verifier.build_prompt(
            cell_line=cell_line, pert=pert, target_gene=gene,
            reasoning=final_reasoning,
            answer=final_answer,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.generate(messages)
        input_prompt = self._format_input_prompt(system_prompt, user_prompt)
        return self.target_verifier.parse_response(response), input_prompt

    def _run_consistency_checker_with_prompt(self, final_reasoning, final_answer, gene, label):
        """Run consistency checker and return (verdict, input_prompt)."""
        if self.task_type == "task2":
            direction = "upregulated" if label == 1 else "downregulated"
            question = f"Will {gene} be {direction}?"
        else:
            question = ""
            direction = ""

        system_prompt, user_prompt = self.consistency_checker.build_prompt(
            reasoning=final_reasoning,
            answer=final_answer,
            task_type=self.task_type,
            question=question,
            ground_truth_direction=direction,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.generate(messages)
        input_prompt = self._format_input_prompt(system_prompt, user_prompt)
        return self.consistency_checker.parse_response(response), input_prompt

    def _run_logic_checker_with_prompt(self, cell_line, pert, gene, final_reasoning, final_answer):
        """Run logic checker and return (verdict, input_prompt)."""
        if self.task_type == "task1":
            question = f"In {cell_line}, will {gene} be upregulated or downregulated by {pert}?"
        else:
            question = f"Will {gene} be affected by {pert}?"

        system_prompt, user_prompt = self.logic_checker.build_prompt(
            question=question,
            answer=final_answer,
            reasoning=final_reasoning,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.generate(messages)
        input_prompt = self._format_input_prompt(system_prompt, user_prompt)
        return self.logic_checker.parse_response(response), input_prompt
