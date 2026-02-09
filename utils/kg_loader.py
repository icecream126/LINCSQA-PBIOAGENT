"""
Knowledge Graph Loader: Load and process KG data from JSON files
Following the approach from kg_to_prompt.ipynb

Supports the following knowledge graphs:
- Ensembl: Gene descriptions
- UniProt: Protein annotations (functions, interactions)
- GO: Gene Ontology
- Reactome: Metabolic/signaling pathways
- CORUM: Protein complexes
- BioPlex: Protein-protein interactions
- STRING: Protein interactions with evidence
- SIGNOR: Signaling pathway database with directional regulatory relationships
"""
import csv
import json
import re
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional


class KGLoader:
    """Load and process knowledge graph data from JSON files"""
    
    def __init__(self, kg_dir: str = "data/kg", use_uniprot: bool = True):
        """
        Args:
            kg_dir: Directory containing KG JSON files
            use_uniprot: Whether to load uniprot.json (default: True)
        """
        self.kg_dir = Path(kg_dir)
        self.use_uniprot = use_uniprot
        self._load_all_data()
        self._process_graphs()
    
    def _load_all_data(self):
        """Load all KG JSON files"""
        # Ensembl
        with open(self.kg_dir / "ensembl.json") as f:
            ensembl_data = json.load(f)
        self.ensembl_dict = {entry["name"]: entry["description"] for entry in ensembl_data.values()}
        
        # UniProt
        if self.use_uniprot:
            with open(self.kg_dir / "uniprot.json") as f:
                self.uniprot_data = json.load(f)
            self.uniprot_dict = defaultdict(list)
            for entry in self.uniprot_data:
                genes = entry.get("gene", [])
                if isinstance(genes, str):
                    genes = [genes]
                elif not isinstance(genes, list):
                    # Skip if gene is not a string or list
                    continue
                for gene in genes:
                    # Ensure gene is a string (hashable)
                    if isinstance(gene, str):
                        self.uniprot_dict[gene].append(entry)
                # Skip non-string genes (e.g., lists, None, etc.)
            self.uniprot_dict = dict(self.uniprot_dict)
        else:
            self.uniprot_data = []
            self.uniprot_dict = {}
        
        # BioPlex
        with open(self.kg_dir / "bioplex.json") as f:
            self.bioplex_data = json.load(f)
        
        # GO (bipartite graph)
        with open(self.kg_dir / "go.json") as f:
            genes_to_go, go_to_genes, go_to_go = json.load(f)
        
        # Reactome (bipartite graph)
        with open(self.kg_dir / "reactome.json") as f:
            gene_to_reaction, reaction_to_gene = json.load(f)
        # Store reaction_to_gene separately for lookups
        self.reaction_to_gene = reaction_to_gene
        
        # CORUM (bipartite graph)
        with open(self.kg_dir / "corum.json") as f:
            gene_to_complex, complex_to_gene = json.load(f)
        
        # STRING
        with open(self.kg_dir / "string.json") as f:
            self.string_data = json.load(f)
        
        # GO dictionary for term descriptions
        with open(self.kg_dir / "go_dict.json") as f:
            self.go_desc = json.load(f)
        
        # SIGNOR - Signaling pathway database with directional regulatory relationships
        self.signor_data = self._load_signor()
        
        # Combine bipartite graphs
        self.all_graphs = {
            "go": {**genes_to_go, **go_to_genes},
            "reactome": {**gene_to_reaction, **reaction_to_gene},
            "corum": {**gene_to_complex, **complex_to_gene},
            "bioplex": self.bioplex_data,
            "string": self.string_data,
            "signor": self.signor_data,
        }
        
        # GO to English translation
        self.go_to_english = {
            "NOT|acts_upstream_of_or_within": "does not act upstream of or within",
            "NOT|colocalizes_with": "does not colocalize with",
            "NOT|contributes_to": "does not contribute to",
            "NOT|enables": "does not enable",
            "NOT|involved_in": "is not involved in",
            "NOT|is_active_in": "is not active in",
            "NOT|located_in": "is not located in",
            "NOT|part_of": "is not part of",
            "acts_upstream_of": "acts upstream of",
            "acts_upstream_of_negative_effect": "acts upstream of negative effect",
            "acts_upstream_of_or_within": "acts upstream of or within",
            "acts_upstream_of_or_within_negative_effect": "acts upstream of or within negative effect",
            "acts_upstream_of_or_within_positive_effect": "acts upstream of or within positive effect",
            "acts_upstream_of_positive_effect": "acts upstream of positive effect",
            "colocalizes_with": "colocalizes with",
            "contributes_to": "contributes to",
            "enables": "enables",
            "involved_in": "involved in",
            "is_active_in": "is active in",
            "located_in": "is located in",
            "part_of": "is part of",
        }
        
        # STRING evidence types
        self.string_evidence = {
            "database": "database evidence in humans",
            "database_transferred": "database evidence in other animals",
            "experiments": "experimental evidence in humans",
            "experiments_transferred": "experimental evidence in other animals",
            "textmining": "literature evidence in humans",
            "textmining_transferred": "literature evidence in other animals",
        }
    
    def _load_signor(self) -> Dict[str, List[Tuple]]:
        """
        Load SIGNOR signaling pathway database from CSV files.
        
        SIGNOR provides directional regulatory relationships:
        - up-regulates: Gene A increases activity/expression of Gene B
        - down-regulates: Gene A decreases activity/expression of Gene B
        - form complex: Gene A and Gene B form a protein complex
        
        Returns:
            Dictionary mapping gene names to list of (target, effect, mechanism) tuples
        """
        signor_dir = self.kg_dir / "signor"
        nodes_file = signor_dir / "nodes.csv"
        edges_file = signor_dir / "edges.csv"
        
        if not nodes_file.exists() or not edges_file.exists():
            return {}
        
        # Load nodes to map identifiers to gene names
        id_to_name = {}
        try:
            with open(nodes_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get('name', '')
                    identifier = row.get('identifier:ID', '')
                    label = row.get(':LABEL', '')
                    if name and identifier:
                        id_to_name[identifier] = name
                    # Also map by name for proteins
                    if label == 'protein' and name:
                        id_to_name[name] = name
        except Exception as e:
            print(f"Warning: Could not load SIGNOR nodes: {e}")
            return {}
        
        # Load edges to build gene regulatory relationships
        signor_graph = defaultdict(list)
        try:
            with open(edges_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    start_id = row.get(':START_ID', '')
                    end_id = row.get(':END_ID', '')
                    effect_raw = row.get('effect:string[]', '')
                    mechanism_raw = row.get('mechanism:string[]', '')
                    rel_type = row.get(':TYPE', '')
                    score = float(row.get('score:float', 0) or 0)
                    
                    # Skip low-confidence edges
                    if score < 0.3:
                        continue
                    
                    # Get gene names from identifiers
                    source_name = id_to_name.get(start_id, start_id)
                    target_name = id_to_name.get(end_id, end_id)
                    
                    # Skip if source or target looks like a complex ID (starts with SIGNOR-)
                    if source_name.startswith('SIGNOR-') or target_name.startswith('SIGNOR-'):
                        continue
                    
                    # Parse effect (e.g., "up-regulates", "down-regulates", "form complex")
                    effect = effect_raw.strip() if effect_raw else rel_type
                    
                    # Parse mechanism (e.g., "phosphorylation", "binding")
                    mechanism = mechanism_raw.strip() if mechanism_raw else ""
                    
                    # Add to graph (source -> target with effect and mechanism)
                    signor_graph[source_name].append((target_name, effect, mechanism, score))
                    
        except Exception as e:
            print(f"Warning: Could not load SIGNOR edges: {e}")
            return {}
        
        # Sort by score and limit to top relationships per gene
        for gene in signor_graph:
            signor_graph[gene] = sorted(signor_graph[gene], key=lambda x: x[3], reverse=True)[:20]
        
        return dict(signor_graph)
    
    def _process_graphs(self):
        """Remove high-degree nodes (common terms) from graphs"""
        common = defaultdict(list)
        max_degree = 1000
        
        for graph_name, graph in self.all_graphs.items():
            for k, v in graph.items():
                if len(v) > max_degree:
                    common[graph_name].append(k)
        
        # Remove common terms from GO and STRING
        for graph_name in ["go", "string"]:
            for common_term in common[graph_name]:
                if common_term in self.all_graphs[graph_name]:
                    del self.all_graphs[graph_name][common_term]
            self.all_graphs[graph_name] = {
                key: [v for v in val if v not in common[graph_name]]
                for key, val in self.all_graphs[graph_name].items()
            }
    
    def _combine_entries(self, entries, key):
        """Helper function to combine entries (from notebook)"""
        if isinstance(entries, dict):
            entries = [entries]
        vals = [entry[key] for entry in entries if len(entry[key]) > 0]
        vals = [[x] if isinstance(x, str) else x for x in vals]
        if len(vals) == 0:
            return []
        vals = itertools.chain(*vals)
        vals = sorted(set(vals))
        return list(vals)
    
    def ensembl_to_text(self, gene: str) -> List[str]:
        """Convert Ensembl data to text (from notebook)"""
        if gene not in self.ensembl_dict:
            return []
        desc = self.ensembl_dict[gene]
        text = f"Description of gene: {desc}"
        text = text.split("[Source:")[0]  # trim this off
        return [text]
    
    def uniprot_to_text(self, gene: str) -> List[str]:
        """Convert UniProt data to text (from notebook)"""
        if not self.use_uniprot:
            return []
        if gene not in self.uniprot_dict:
            return []
        description = []
        entries = self.uniprot_dict[gene]
        
        proteins = self._combine_entries(entries, "protein")
        if len(proteins) > 0:
            description.append(f"Gene products: {', '.join(proteins)}")
        
        function = self._combine_entries(entries, "function")
        if len(function) > 0:
            description.append(f"Functions: {', '.join(function)}")
        
        subunit = self._combine_entries(entries, "subunit")
        if len(subunit) > 0:
            description.append(f"Quaternary structure: {' '.join(subunit)}")
        
        interaction = self._combine_entries(entries, "interaction")
        if len(interaction) > 0:
            description.append(f"Interacts with: {', '.join(interaction)}")
        
        description = [re.sub(r" \(PubMed.*\)", "", d) for d in description]
        return description
    
    def translate_go_rel(self, rel: str) -> str:
        """Translate GO relationship to English"""
        if rel in self.go_to_english:
            return self.go_to_english[rel]
        elif rel in self.go_desc:
            return self.go_desc[rel]
        return rel
    
    def translate_go(self, g1: str, rels: List[Tuple]) -> List[str]:
        """Translate GO relationships to text (from notebook)"""
        desc = []
        for rel in rels:
            go_term, rel_type = rel
            desc.append(f"{g1} {self.translate_go_rel(rel_type)} {self.translate_go_rel(go_term)}.")
        desc = [
            d for d in desc
            if "molecular_function" not in d
            and "biological_process" not in d
            and "protein_binding" not in d
        ]
        return desc
    
    def translate_reactome(self, g1: str, rels: List[Tuple]) -> List[str]:
        """Translate Reactome relationships to text (from notebook)"""
        loc_to_g2s = defaultdict(list)
        for rel in rels:
            # rel is (reaction_name, location) from gene_to_reaction
            # Convert to tuple if it's a list
            if isinstance(rel, list):
                rel = tuple(rel)
            
            if len(rel) >= 2:
                reaction_name, loc1 = rel[0], rel[1]
                # Get all genes for this reaction
                if reaction_name in self.reaction_to_gene:
                    for gene_rel in self.reaction_to_gene[reaction_name]:
                        # gene_rel is [gene, location]
                        if isinstance(gene_rel, (list, tuple)) and len(gene_rel) >= 1:
                            g2 = gene_rel[0]
                            # Use location from gene_rel if available, otherwise use loc1
                            loc = gene_rel[1] if len(gene_rel) > 1 else loc1
                            if g2 != g1:  # Don't include self
                                loc_to_g2s[loc].append(g2)
        
        desc = []
        for loc1, g2s in loc_to_g2s.items():
            # Remove duplicates while preserving order
            unique_g2s = []
            seen = set()
            for g2 in g2s:
                if g2 not in seen:
                    seen.add(g2)
                    unique_g2s.append(g2)
            if unique_g2s:
                desc.append(f"In the {loc1}, {g1} enables {', '.join(unique_g2s)}")
        return desc
    
    def translate_bioplex(self, g1: str, rels: List[Tuple]) -> List[str]:
        """Translate BioPlex relationships to text (from notebook)"""
        cell_to_g2s = defaultdict(list)
        for rel in rels:
            g2, celltype = rel
            cell_to_g2s[celltype].append(g2)
        desc = []
        for celltype, g2s in cell_to_g2s.items():
            desc.append(f"In {celltype} cells, {g1} may form a complex with {', '.join(g2s)}")
        return desc
    
    def translate_string(self, g1: str, rels: List[Tuple]) -> List[str]:
        """Translate STRING relationships to text (from notebook)"""
        rel_to_g2s = defaultdict(list)
        for rel in rels:
            g2, rel1 = rel
            if isinstance(rel1, (list, tuple)):
                rel1_str = ", ".join([self.string_evidence.get(e, e) for e in rel1])
            else:
                rel1_str = self.string_evidence.get(rel1, rel1)
            rel_to_g2s[rel1_str].append(g2)
        desc = []
        for rel1, g2s in rel_to_g2s.items():
            desc.append(f"Based on evidence from {rel1}, {g1} may physically interact with {', '.join(g2s)}.")
        return desc
    
    def translate_corum(self, g1: str, rels: List[Tuple]) -> List[str]:
        """Translate CORUM relationships to text (from notebook)"""
        desc = []
        for rel in rels:
            complex1, celltype1 = rel
            desc.append(f"{g1} is a member of {complex1} in {celltype1}.")
        return desc
    
    def translate_signor(self, g1: str, rels: List[Tuple]) -> List[str]:
        """
        Translate SIGNOR regulatory relationships to text.
        
        SIGNOR provides directional regulatory information which is particularly
        valuable for predicting gene up/down-regulation.
        
        Args:
            g1: Source gene name
            rels: List of (target, effect, mechanism, score) tuples
            
        Returns:
            List of descriptive strings
        """
        desc = []
        for rel in rels:
            if len(rel) >= 3:
                target, effect, mechanism = rel[0], rel[1], rel[2]
                if mechanism and mechanism.lower() not in ['', 'none', 'unknown']:
                    desc.append(f"{g1} {effect} {target} via {mechanism}.")
                else:
                    desc.append(f"{g1} {effect} {target}.")
        return desc
    
    def get_gene_descriptions(self, gene: str, max_items: int = 50) -> List[str]:
        """
        Get all descriptions for a gene following notebook approach
        
        Args:
            gene: Gene name
            max_items: Maximum number of description items
            
        Returns:
            List of description strings
        """
        desc = set()
        
        # Priority 1: Ensembl and UniProt (always included first)
        desc.update(self.ensembl_to_text(gene))
        if self.use_uniprot:
            desc.update(self.uniprot_to_text(gene))
        
        # Priority 2: Graph-based sources (SIGNOR first as it has directional regulation info)
        for graph_name in ["signor", "reactome", "corum", "go", "bioplex", "string"]:
            graph = self.all_graphs[graph_name]
            if gene not in graph:
                continue
            
            rels = graph[gene]
            
            # Tuple mashing (from notebook)
            # Convert nested lists to tuples to make them hashable
            if isinstance(rels, list) and len(rels) > 0:
                if graph_name == "string":
                    # STRING: [(gene, [evidence_types])]
                    rels = [(r[0], tuple(r[1]) if isinstance(r[1], list) else r[1]) for r in rels]
                else:
                    # For other graphs, convert lists to tuples
                    # Handle both single-level and nested lists
                    converted_rels = []
                    for r in rels:
                        if isinstance(r, list):
                            # Convert list to tuple, handling nested lists
                            if len(r) > 0 and isinstance(r[0], list):
                                # Nested list: [[a, b], [c, d]] -> keep as is for now
                                converted_rels.append(tuple(r) if len(r) == 2 else tuple(r))
                            else:
                                # Single level list: [a, b] -> (a, b)
                                converted_rels.append(tuple(r))
                        else:
                            converted_rels.append(r)
                    rels = converted_rels
            
            # Filtering (from notebook)
            if len(rels) > 50 and graph_name == "string":
                rels = [r for r in rels if len(r[1]) > 1]
            if len(rels) > 50 and graph_name == "bioplex":
                continue
            
            # Translate
            if graph_name == "go":
                translated = self.translate_go(gene, rels)
            elif graph_name == "reactome":
                translated = self.translate_reactome(gene, rels)
            elif graph_name == "corum":
                translated = self.translate_corum(gene, rels)
            elif graph_name == "bioplex":
                translated = self.translate_bioplex(gene, rels)
            elif graph_name == "string":
                translated = self.translate_string(gene, rels)
            elif graph_name == "signor":
                translated = self.translate_signor(gene, rels)
            else:
                translated = []
            
            # Ensure all translated items are strings
            for item in translated:
                if isinstance(item, str):
                    desc.add(item)
                elif isinstance(item, (list, tuple)):
                    # Convert list/tuple to string representation
                    desc.add(str(item))
                else:
                    desc.add(str(item))
            
            if len(desc) > max_items:
                break
        
        # Convert to list and limit
        desc_list = list(desc)
        return desc_list[:max_items]

