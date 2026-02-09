# GUIDELINE.md: Progressive Reasoning Context Generation for Integration Agent

## Overview

This guideline describes how to generate structured progressive reasoning context for the Integration Agent. The goal is to enable effective knowledge transfer from HIGH-CONFIDENCE (easy) gene predictions to LOW-CONFIDENCE (hard) gene predictions within the same perturbation context.

---

## 1. Core Concepts

### 1.1 Progressive Reasoning Philosophy

The key insight is that genes affected by the same perturbation share causal structure. When we predict gene regulation responses:

1. **Easy genes** have direct, well-documented mechanistic links to the perturbation
2. **Hard genes** have indirect, complex, or poorly-annotated relationships
3. **Knowledge transfer**: Validated reasoning from easy genes can anchor predictions for hard genes

### 1.2 Why Difficulty-Aware Ordering Matters

| Ordering Strategy | Behavior | Expected Outcome |
|-------------------|----------|------------------|
| Random | No structure in context accumulation | Reasoning may be inconsistent |
| Difficulty-Aware | Easy → Hard progression | High-confidence reasoning anchors uncertain cases |

---

## 2. Difficulty Score Definition

### 2.1 Composite Difficulty Score Formula
```
difficulty_score = self_consistency_score × biological_relevance_score
```

Where:
- **Range**: 0.0 (hardest) to 1.0 (easiest)
- **Higher score** = Easier to predict
- **Lower score** = Harder to predict

### 2.2 Component 1: Self-Consistency Score

Measures prediction stability across multiple stochastic LLM trials.
```python
def calculate_self_consistency(cell_line, perturbation, gene, n_trials=5):
    """
    Ask the base LLM the same question n_trials times.
    Returns the proportion of the majority answer.
    """
    predictions = []
    for _ in range(n_trials):
        pred = base_llm.predict(
            f"In {cell_line}, when treated with {perturbation}, "
            f"will {gene} be upregulated or downregulated?"
        )
        predictions.append(pred)  # "up" or "down"
    
    majority_count = max(predictions.count("up"), predictions.count("down"))
    return majority_count / n_trials  # Range: 0.5 to 1.0
```

**Interpretation**:
- 1.0 (5/5 agreement): Very consistent, likely easy
- 0.8 (4/5 agreement): Moderately consistent
- 0.6 (3/5 agreement): Inconsistent, likely hard
- 0.5 (random): Maximum uncertainty

### 2.3 Component 2: Biological Relevance Score

Measures the strength of known biological connection between perturbation target and gene.
```python
def calculate_biological_relevance(perturbation_target, gene):
    """
    Query STRING database for interaction score.
    Returns normalized score between 0 and 1.
    """
    # STRING combined_score ranges from 0 to 1000
    string_score = string_db.get_interaction_score(perturbation_target, gene)
    
    # Normalize to 0-1 range
    return string_score / 1000.0
```

**Alternative sources** (if STRING score unavailable):
- Pathway co-membership score (Reactome, KEGG)
- Literature co-occurrence score
- Knowledge graph path distance

### 2.4 Difficulty Categories

| Score Range | Category | Description |
|-------------|----------|-------------|
| 0.75 - 1.00 | EASY | Direct pathway link, high LLM consistency |
| 0.50 - 0.74 | MEDIUM | Indirect link or moderate consistency |
| 0.25 - 0.49 | HARD | Weak/no direct link, low consistency |
| 0.00 - 0.24 | VERY HARD | No known link, near-random LLM predictions |

---

## 3. Key Reasoning Generation

### 3.1 Purpose

Key Reasoning summarizes WHY a gene was predicted in a certain direction. It should be:
- **Mechanistic**: Describes the causal chain from perturbation to gene expression change
- **Concise**: 2-4 bullet points maximum
- **Specific**: References actual pathway/molecular interactions

### 3.2 Generation Process
```python
def generate_key_reasoning(agent_outputs, prediction, gene_info):
    """
    Synthesize key reasoning from agent outputs.
    
    Args:
        agent_outputs: Dict with context, mechanism, network agent results
        prediction: "upregulated" or "downregulated"
        gene_info: Gene metadata and pathway annotations
    
    Returns:
        List of 2-4 key reasoning bullet points
    """
    prompt = f"""
    Based on the following agent analyses, extract 2-4 KEY REASONING points
    that explain why {gene_info['symbol']} is {prediction}.
    
    Context Agent: {agent_outputs['context']}
    Mechanism Agent: {agent_outputs['mechanism']}
    Network Agent: {agent_outputs['network']}
    
    Requirements:
    1. Each point should describe ONE step in the causal chain
    2. Use format: [Cause] → [Effect]
    3. Be specific to this gene, not generic HDAC effects
    4. Prioritize direct mechanistic links over indirect ones
    
    Output as JSON list of strings.
    """
    return llm.generate(prompt)
```

### 3.3 Key Reasoning Examples

**Good Example (Specific)**:
```
Key Reasoning for TP53:
- HDAC3 directly deacetylates TP53 protein, reducing its stability
- ACY-1215 inhibits HDAC3 → TP53 acetylation increases → TP53 stabilized
- Stabilized TP53 activates its own transcription via autoregulatory loop
- Result: TP53 mRNA increases
```

**Bad Example (Too Generic)**:
```
Key Reasoning for TP53:
- HDAC inhibition increases acetylation
- Acetylation activates transcription
- TP53 is upregulated
```

### 3.4 Key Reasoning Template
```
Key Reasoning for {GENE}:
- [Perturbation] [action verb] [direct target]: {mechanistic detail}
- [Intermediate step]: {how signal propagates}
- [Gene-specific effect]: {why THIS gene responds this way}
- Result: {GENE} mRNA {increases/decreases}
```

---

## 4. Transferable Insight Generation

### 4.1 Purpose

Transferable Insight bridges the gap between an anchor gene and the target gene. It explicitly states:
1. What mechanistic principle can be applied to other genes
2. Under what conditions this principle applies
3. How to check if it applies to the target gene

### 4.2 Generation Process
```python
def generate_transferable_insight(
    anchor_gene: str,
    anchor_reasoning: List[str],
    target_gene: str,
    target_gene_info: dict,
    string_similarity: float
):
    """
    Generate transferable insight from anchor to target gene.
    """
    prompt = f"""
    You are analyzing how reasoning for {anchor_gene} might transfer to {target_gene}.
    
    Anchor Gene: {anchor_gene}
    Anchor Key Reasoning:
    {format_as_bullets(anchor_reasoning)}
    
    Target Gene: {target_gene}
    Target Gene Info:
    - Function: {target_gene_info['function']}
    - Pathways: {target_gene_info['pathways']}
    - Cellular location: {target_gene_info['location']}
    
    STRING Similarity Score: {string_similarity}
    
    Generate 2-3 TRANSFERABLE INSIGHTS using this format:
    
    "→ IF [condition about target gene], THEN [expected outcome based on anchor reasoning]"
    
    Requirements:
    1. Each insight must be TESTABLE against target gene's known properties
    2. Be specific about what property of the target gene would make the transfer valid
    3. Include at least one insight about shared pathway membership
    4. Include at least one insight about regulatory mechanism similarity
    
    Output as JSON list of strings.
    """
    return llm.generate(prompt)
```

### 4.3 Transferable Insight Examples

**Good Example (Testable, Specific)**:
```
Transferable Insights from TP53 to ARL4C:

→ IF ARL4C has acetylation-responsive promoter elements (e.g., CBP/p300 binding sites), 
  THEN expect upregulation similar to TP53

→ IF ARL4C is transcriptionally regulated by TP53 (check for p53 response elements), 
  THEN ARL4C may be a downstream target activated by stabilized TP53

→ IF ARL4C participates in cell cycle regulation pathways shared with TP53 
  (both show nuclear localization), THEN coordinated regulation is likely
```

**Bad Example (Vague, Not Testable)**:
```
Transferable Insights from TP53 to ARL4C:

→ ARL4C might be upregulated like TP53
→ Both genes could respond to HDAC inhibition similarly
→ Acetylation might affect ARL4C
```

### 4.4 Transferable Insight Template
```
Transferable Insights from {ANCHOR_GENE} to {TARGET_GENE}:

→ IF {TARGET_GENE} {testable_condition_1},
  THEN {expected_outcome_based_on_anchor}
  [Relevance: {why this matters}]

→ IF {TARGET_GENE} {testable_condition_2},
  THEN {expected_outcome_based_on_anchor}
  [Relevance: {why this matters}]

→ IF {shared_pathway_condition},
  THEN {coordinated_regulation_expectation}
  [Relevance: {pathway name and function}]
```

---

## 5. Complete Prompt Structure

### 5.1 Progressive Reasoning Context Section
```
== Progressive Reasoning Context ==
The following genes have been analyzed in CONFIDENCE ORDER (highest first).
Their validated reasoning provides mechanistic anchors for predicting {TARGET_GENE}.

┌─────────────────────────────────────────────────────────────────────────┐
│ ANCHOR PREDICTION {N}: {GENE_SYMBOL}                                    │
│ Prediction: {UPREGULATED/DOWNREGULATED}                                 │
│ Confidence: {XX}% | Difficulty: {EASY/MEDIUM/HARD} (Score: {X.XX})     │
│ STRING similarity to {TARGET_GENE}: {X.XXX}                            │
├─────────────────────────────────────────────────────────────────────────┤
│ Key Reasoning:                                                          │
│ • {reasoning_point_1}                                                   │
│ • {reasoning_point_2}                                                   │
│ • {reasoning_point_3}                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ Transferable Insights for {TARGET_GENE}:                               │
│ → {insight_1}                                                           │
│ → {insight_2}                                                           │
│ → {insight_3}                                                           │
└─────────────────────────────────────────────────────────────────────────┘

[Repeat for each anchor gene, ordered by confidence]
```

### 5.2 Target Gene Section
```
== Current Target: {TARGET_GENE} ==
Difficulty: {CATEGORY} (Score: {X.XX})

Why this prediction is {EASY/MEDIUM/HARD}:
- Self-consistency: {X/5} trials agreed on same direction
- STRING score with perturbation target: {X.XXX}
- Direct pathway annotation: {EXISTS/MISSING}
- {Additional difficulty factor if applicable}

Specific Challenges:
- {challenge_1}: {description}
- {challenge_2}: {description}
```

### 5.3 Task Instruction Section
```
== YOUR TASK ==

1. ANALYZE ANCHOR RELEVANCE
   - For each anchor gene, evaluate whether its transferable insights apply to {TARGET_GENE}
   - Check the stated conditions against {TARGET_GENE}'s known properties
   - Document which insights are applicable and which are not

2. SYNTHESIZE PREDICTION
   - If transferable insights apply: Use anchor reasoning as foundation
   - If insights don't apply: Rely more heavily on agent evidence and NN prediction
   - Explain your reasoning chain explicitly

3. DIFFERENTIATE YOUR REASONING
   - Your reasoning must be SPECIFIC to {TARGET_GENE}
   - Do NOT simply copy anchor gene reasoning
   - Explain WHY {TARGET_GENE} follows (or doesn't follow) the same pattern

4. CONFIDENCE-WEIGHTED DECISION
   - NN Confidence >80%: Strongly trust NN prediction
   - NN Confidence 60-80%: Validate with biological reasoning
   - NN Confidence <60%: Biological reasoning may override

5. OUTPUT REQUIREMENTS
   - canonical_reasoning: Direct MoA-based reasoning specific to {TARGET_GENE}
   - counterfactual_reasoning: How could the opposite prediction be justified?
   - reasoning: Final synthesis explaining your decision
   - answer: upregulated OR downregulated
```

---

## 6. Implementation Pseudocode

### 6.1 Main Pipeline
```python
def progressive_reasoning_pipeline(cell_line, perturbation, genes, moa_info):
    """
    Main pipeline for progressive reasoning across genes.
    """
    # Step 1: Calculate difficulty scores for all genes
    difficulty_scores = {}
    for gene in genes:
        consistency = calculate_self_consistency(cell_line, perturbation, gene)
        relevance = calculate_biological_relevance(moa_info['target'], gene)
        difficulty_scores[gene] = consistency * relevance
    
    # Step 2: Sort genes by difficulty (easy first)
    sorted_genes = sorted(genes, key=lambda g: difficulty_scores[g], reverse=True)
    
    # Step 3: Progressive prediction
    predictions = {}
    reasoning_traces = {}
    
    for i, gene in enumerate(sorted_genes):
        # Get anchor genes (previously predicted, higher confidence)
        anchor_genes = sorted_genes[:i]
        
        # Build progressive context
        progressive_context = build_progressive_context(
            anchor_genes=anchor_genes,
            predictions=predictions,
            reasoning_traces=reasoning_traces,
            target_gene=gene,
            difficulty_scores=difficulty_scores
        )
        
        # Run agents
        agent_outputs = run_specialized_agents(cell_line, perturbation, gene, moa_info)
        
        # Run integration agent with progressive context
        result = integration_agent(
            cell_line=cell_line,
            perturbation=perturbation,
            gene=gene,
            progressive_context=progressive_context,
            agent_outputs=agent_outputs,
            difficulty_score=difficulty_scores[gene]
        )
        
        # Store results
        predictions[gene] = result['answer']
        reasoning_traces[gene] = {
            'key_reasoning': extract_key_reasoning(result),
            'full_reasoning': result['reasoning'],
            'confidence': result.get('confidence', calculate_confidence(result))
        }
    
    return predictions, reasoning_traces


def build_progressive_context(anchor_genes, predictions, reasoning_traces, 
                              target_gene, difficulty_scores):
    """
    Build structured progressive context for integration agent.
    """
    context_blocks = []
    
    for i, anchor in enumerate(anchor_genes[-3:]):  # Use top 3 most recent anchors
        # Get STRING similarity
        string_sim = get_string_similarity(anchor, target_gene)
        
        # Generate transferable insights
        transferable_insights = generate_transferable_insight(
            anchor_gene=anchor,
            anchor_reasoning=reasoning_traces[anchor]['key_reasoning'],
            target_gene=target_gene,
            target_gene_info=get_gene_info(target_gene),
            string_similarity=string_sim
        )
        
        # Build context block
        block = format_anchor_block(
            gene=anchor,
            prediction=predictions[anchor],
            confidence=reasoning_traces[anchor]['confidence'],
            difficulty_score=difficulty_scores[anchor],
            key_reasoning=reasoning_traces[anchor]['key_reasoning'],
            transferable_insights=transferable_insights,
            string_similarity=string_sim,
            target_gene=target_gene
        )
        context_blocks.append(block)
    
    # Build target gene difficulty section
    target_difficulty = format_target_difficulty(
        gene=target_gene,
        difficulty_score=difficulty_scores[target_gene],
        gene_info=get_gene_info(target_gene)
    )
    
    return {
        'anchor_blocks': context_blocks,
        'target_difficulty': target_difficulty,
        'task_instruction': generate_task_instruction(target_gene)
    }
```

### 6.2 Helper Functions
```python
def get_difficulty_category(score):
    """Convert numeric score to category label."""
    if score >= 0.75:
        return "EASY"
    elif score >= 0.50:
        return "MEDIUM"
    elif score >= 0.25:
        return "HARD"
    else:
        return "VERY HARD"


def format_anchor_block(gene, prediction, confidence, difficulty_score,
                        key_reasoning, transferable_insights, 
                        string_similarity, target_gene):
    """Format a single anchor gene block."""
    category = get_difficulty_category(difficulty_score)
    
    block = f"""
┌─────────────────────────────────────────────────────────────────────────┐
│ ANCHOR PREDICTION: {gene}                                               │
│ Prediction: {prediction.upper()}                                        │
│ Confidence: {confidence:.0%} | Difficulty: {category} (Score: {difficulty_score:.2f})│
│ STRING similarity to {target_gene}: {string_similarity:.3f}            │
├─────────────────────────────────────────────────────────────────────────┤
│ Key Reasoning:                                                          │
"""
    for point in key_reasoning:
        block += f"│ • {point}\n"
    
    block += """├─────────────────────────────────────────────────────────────────────────┤
│ Transferable Insights for {target_gene}:                               │
"""
    for insight in transferable_insights:
        block += f"│ → {insight}\n"
    
    block += "└─────────────────────────────────────────────────────────────────────────┘"
    
    return block


def format_target_difficulty(gene, difficulty_score, gene_info):
    """Format the target gene difficulty section."""
    category = get_difficulty_category(difficulty_score)
    
    # Determine specific challenges based on score components
    challenges = []
    if gene_info.get('string_score', 0) < 0.3:
        challenges.append("No direct pathway annotation linking to perturbation target")
    if gene_info.get('self_consistency', 0) < 0.7:
        challenges.append("Low prediction consistency across LLM trials")
    if not gene_info.get('has_moa_annotation', False):
        challenges.append("Gene function not obviously related to perturbation mechanism")
    
    return f"""
== Current Target: {gene} ==
Difficulty: {category} (Score: {difficulty_score:.2f})

Why this prediction is {category}:
- Self-consistency: {gene_info.get('consistency_trials', 'N/A')}
- STRING score with perturbation target: {gene_info.get('string_score', 'N/A'):.3f}
- Direct pathway annotation: {'EXISTS' if gene_info.get('has_pathway', False) else 'MISSING'}

Specific Challenges:
{chr(10).join(f'• {c}' for c in challenges) if challenges else '• None identified'}
"""
```

---

## 7. Validation Checklist

Before deploying the updated prompts, verify:

### 7.1 Key Reasoning Quality
- [ ] Each point describes a specific mechanistic step
- [ ] Uses [Cause] → [Effect] format
- [ ] Specific to the gene, not generic perturbation effects
- [ ] 2-4 points maximum

### 7.2 Transferable Insight Quality
- [ ] Each insight is testable (IF condition, THEN outcome)
- [ ] Conditions reference specific gene properties
- [ ] At least one pathway-based insight
- [ ] At least one mechanism-based insight

### 7.3 Difficulty Score Validity
- [ ] Score correlates with actual prediction accuracy
- [ ] Easy genes have higher accuracy than hard genes
- [ ] Categories (EASY/MEDIUM/HARD) are well-separated

### 7.4 Progressive Context Effectiveness
- [ ] Random order performs worse than difficulty-aware order
- [ ] Integration agent explicitly references anchor reasoning
- [ ] No history leakage (copying predictions without justification)

---

## 8. Ablation Study Configuration

### 8.1 Experimental Conditions

| Condition | Ordering | Key Reasoning | Transferable Insights | Difficulty Info |
|-----------|----------|---------------|----------------------|-----------------|
| Baseline (No History) | N/A | ✗ | ✗ | ✗ |
| Random Order | Random | ✗ | ✗ | ✗ |
| Random + Reasoning | Random | ✓ | ✗ | ✗ |
| Difficulty Order | Easy→Hard | ✗ | ✗ | ✗ |
| Difficulty + Reasoning | Easy→Hard | ✓ | ✗ | ✗ |
| Full Progressive | Easy→Hard | ✓ | ✓ | ✓ |

### 8.2 Expected Results Pattern
```
Accuracy (expected):

Full Progressive     ████████████████████  (highest)
Difficulty+Reasoning ███████████████████
Difficulty Order     ██████████████████
Random+Reasoning     █████████████████
Random Order         ████████████████
Baseline             ███████████████       (lowest)
```

### 8.3 Metrics to Report

1. **Overall Accuracy**: AUROC across all genes
2. **Stratified Accuracy**: AUROC by difficulty category (EASY/MEDIUM/HARD)
3. **Transfer Effectiveness**: Accuracy improvement on HARD genes when anchored by EASY genes
4. **History Utilization Rate**: % of responses that explicitly reference anchor reasoning

---

## 9. Common Pitfalls to Avoid

### 9.1 History Leakage
**Problem**: Model copies anchor prediction without new justification
**Solution**: Task instruction explicitly requires differentiated reasoning

### 9.2 Generic Reasoning
**Problem**: Key reasoning uses generic "acetylation → transcription" logic
**Solution**: Require gene-specific mechanistic details

### 9.3 Untestable Insights
**Problem**: Transferable insights are vague ("might be similar")
**Solution**: Require IF-THEN format with testable conditions

### 9.4 Difficulty Score Miscalibration
**Problem**: Easy/Hard labels don't correlate with actual difficulty
**Solution**: Validate scores against prediction accuracy on held-out data
