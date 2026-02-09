"""
Scientist Agent classes for Multi-Scientist Multi-Judge Pipeline.

Contains:
- ContextAgent: Analyzes cell line background and genomic context
- MechanismAgent: Identifies the first biochemical event caused by the perturbation
- NetworkAgent: Traces signaling and regulatory pathways
- IntegrationAgent: Aggregates outputs from all scientist agents and GAT prediction
"""

import logging
from typing import Dict, Optional, Tuple

from utils.prompt import return_prompt
from utils.knowledge import filter_moa_from_depmap_context
from utils.config import get_moa_knowledge, get_compound_knowledge

logger = logging.getLogger(__name__)


class ContextAgent:
    """Analyzes cell line background and genomic context."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.system_prompt = return_prompt("context_agent_system_prompt")
        self.user_prompt_template = return_prompt("context_agent_user_prompt")

    def build_prompt(
        self,
        cell_line: str,
        pert: str,
        target_gene: str,
        cell_line_info: str = "",
        kg_context: str = "",
        depmap_context: str = "",
        compound: str = "",
    ) -> Tuple[str, str]:
        """Build system and user prompts."""
        user_prompt = self.user_prompt_template.format(
            cell_line=cell_line, pert_or_moa=pert, target_gene=target_gene
        )

        # Add cell line information
        if cell_line_info:
            user_prompt += f"\n\n### Cell Line Information\n{cell_line_info}"

        # Add MoA domain knowledge
        moa_knowledge_str = get_moa_knowledge(pert)
        if moa_knowledge_str:
            user_prompt += f"\n\n### MoA Domain Knowledge\n{moa_knowledge_str}"

        # Add compound domain knowledge
        compound_knowledge_str = get_compound_knowledge(compound)
        if compound_knowledge_str:
            user_prompt += f"\n\n### Compound Domain Knowledge\n{compound_knowledge_str}"

        # Add KG context (includes gene info and pathway context)
        if kg_context:
            user_prompt += f"\n\n### Knowledge Graph Context\n{kg_context}"

        # Add DepMap context
        if depmap_context:
            filtered_depmap = filter_moa_from_depmap_context(depmap_context)
            if filtered_depmap.strip():
                user_prompt += f"\n\n### DepMap Biological Context\n{filtered_depmap}"
        return self.system_prompt, user_prompt


class MechanismAgent:
    """Identifies the first biochemical event caused by the perturbation."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.system_prompt = return_prompt("mechanism_agent_system_prompt")
        self.user_prompt_template = return_prompt("mechanism_agent_user_prompt")

    def build_prompt(
        self, 
        pert: str, 
        target_gene: str, 
        compound: str = "",
        cell_line_info: str = "",
        kg_context: str = "",
    ) -> Tuple[str, str]:
        """Build system and user prompts."""
        user_prompt = self.user_prompt_template.format(
            pert_or_moa=pert, target_gene=target_gene, drug_name=compound
        )

        # Add cell line information
        if cell_line_info:
            user_prompt += f"\n\n### Cell Line Information\n{cell_line_info}"

        # Add MoA domain knowledge
        moa_knowledge_str = get_moa_knowledge(pert)
        if moa_knowledge_str:
            user_prompt += f"\n\n### MoA Domain Knowledge\n{moa_knowledge_str}"

        # Add compound domain knowledge
        compound_knowledge_str = get_compound_knowledge(compound)
        if compound_knowledge_str:
            user_prompt += f"\n\n### Compound Domain Knowledge\n{compound_knowledge_str}"

        # Add KG context (includes gene info and pathway context)
        if kg_context:
            user_prompt += f"\n\n### Knowledge Graph Context\n{kg_context}\n\nIMPORTANT: Use pathway notation format (Gene) -(relationship)-> (Entity)."
        
        return self.system_prompt, user_prompt


class NetworkAgent:
    """Traces signaling and regulatory pathways."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.system_prompt = return_prompt("network_agent_system_prompt")
        self.user_prompt_template = return_prompt("network_agent_user_prompt")

    def build_prompt(
        self, 
        pert: str, 
        target_gene: str, 
        cell_line_info: str = "",
        kg_context: str = "",
        compound: str = "",
    ) -> Tuple[str, str]:
        """Build system and user prompts."""
        user_prompt = self.user_prompt_template.format(
            pert_target=pert, target_gene=target_gene
        )

        # Add cell line information
        if cell_line_info:
            user_prompt += f"\n\n### Cell Line Information\n{cell_line_info}"

        # Add MoA domain knowledge
        moa_knowledge_str = get_moa_knowledge(pert)
        if moa_knowledge_str:
            user_prompt += f"\n\n### MoA Domain Knowledge\n{moa_knowledge_str}"

        # Add compound domain knowledge
        compound_knowledge_str = get_compound_knowledge(compound)
        if compound_knowledge_str:
            user_prompt += f"\n\n### Compound Domain Knowledge\n{compound_knowledge_str}"

        # Add KG context (includes gene info and pathway context)
        if kg_context:
            user_prompt += f"\n\n### Knowledge Graph Context\n{kg_context}\n\nIMPORTANT: Use pathway notation format (Gene) -(relationship)-> (Entity)."
        
        return self.system_prompt, user_prompt


class IntegrationAgent:
    """Aggregates outputs from all scientist agents and GAT prediction."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        # Task1 prompts (with GAT)
        self.system_prompt = return_prompt("integration_agent_gat_system_prompt")
        self.user_prompt_template = return_prompt("integration_agent_gat_user_prompt")
        # Task2 prompts (without GAT, simpler reasoning)
        self.system_prompt_task2 = return_prompt("integration_agent_task2_system_prompt")
        self.user_prompt_template_task2 = return_prompt("integration_agent_task2_user_prompt")

    def build_prompt(
        self,
        cell_line: str,
        pert: str,
        target_gene: str,
        context_reasoning: str,
        mechanism_reasoning: str,
        network_reasoning: str,
        gat_context: str,
        history_context: str = "",
        cell_line_info: str = "",
        kg_context: str = "",
        judge_feedback: str = "",
        compound: str = "",
        candidate_moa: str = "",
        task_type: str = "task1",
        label: int = 0,
    ) -> Tuple[str, str]:
        """Build system and user prompts.

        For task1: Uses GAT-enhanced prompts with canonical/counterfactual reasoning.
        For task2: Uses simplified prompts without GAT, asking for biologically reasonable reasoning.
        """
        # Format judge feedback section if present
        feedback_section = ""
        if judge_feedback:
            feedback_section = f"\n\n[Previous Attempt Feedback - Address These Issues]\n{judge_feedback}\n"

        # Extract MoA name from pert (e.g., "KRAS inhibitor(ARS1620)" -> "KRAS inhibitor")
        # or use candidate_moa if provided
        moa_name = candidate_moa
        if not moa_name and "(" in pert:
            moa_name = pert.split("(")[0].strip()
        elif not moa_name:
            moa_name = pert

        # Build MoA and compound domain knowledge sections
        moa_knowledge_str = get_moa_knowledge(moa_name)
        compound_knowledge_str = get_compound_knowledge(compound)

        # Append domain knowledge to cell_line_info if available
        enhanced_cell_line_info = cell_line_info
        if moa_knowledge_str:
            enhanced_cell_line_info += f"\n\n### MoA Domain Knowledge\n{moa_knowledge_str}"
        if compound_knowledge_str:
            enhanced_cell_line_info += f"\n\n### Compound Domain Knowledge\n{compound_knowledge_str}"

        if task_type == "task2":
            # Task2: Use simplified prompts without GAT
            direction_asked = "upregulated" if label == 1 else "downregulated"

            user_prompt = self.user_prompt_template_task2.format(
                target_gene=target_gene,
                direction_asked=direction_asked,
                pert_or_moa=pert,
                context_reasoning=context_reasoning,
                mechanism_reasoning=mechanism_reasoning,
                network_reasoning=network_reasoning,
                history_context=history_context if history_context else "",
                cell_line_info=enhanced_cell_line_info,
                kg_context=kg_context,
            )

            system_prompt = self.system_prompt_task2
        else:
            # Task1: Use GAT-enhanced prompts
            user_prompt = self.user_prompt_template.format(
                cell_line=cell_line,
                target_gene=target_gene,
                pert_or_moa=pert,
                context_reasoning=context_reasoning,
                mechanism_reasoning=mechanism_reasoning,
                network_reasoning=network_reasoning,
                gat_context=gat_context,
                history_context=history_context,
                cell_line_info=enhanced_cell_line_info,
                kg_context=kg_context,
            )
            system_prompt = self.system_prompt

        # Append feedback section if present
        if feedback_section:
            user_prompt += feedback_section

        return system_prompt, user_prompt
