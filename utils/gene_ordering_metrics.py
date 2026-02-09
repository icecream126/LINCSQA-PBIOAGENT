import time
import pandas as pd
import networkx as nx
import numpy as np

metric_descriptions = {
        'STRINGdb_score': """This function uses the STRING database and Dijkstra's algorithm to calculate and rank the interaction score between a knockdown gene and a list of target genes. Based on the STRING database, values close to or above 1 suggest strong interactions, while those close to 0 indicate weak or no interaction.""",
        'GO_semantic_similarity': """GO semantic similarity measures the functional relatedness between two genes based on their shared Gene Ontology (GO) terms. It uses Jaccard similarity to compare the overlap of GO terms (Biological Process, Cellular Component, and Molecular Function) between genes. Values range from 0.0 (no shared GO terms) to 1.0 (identical GO term sets), with higher values indicating greater functional similarity."""
    }  
def return_metric_descriptions(metric_name: str):
    return metric_descriptions[metric_name]

class StringDBLocal:
    def __init__(
        self,
        links_path="data/metadata/STRINGdb/9606.protein.links.v12.0.txt",
        info_path="data/metadata/STRINGdb/9606.protein.info.v12.0.txt",
        min_confidence_score=400
    ):
        print("Loading local STRINGdb...")
        start = time.time()
        self.graph = self.load_string_local(links_path, info_path, min_confidence_score)
        print(f"STRINGdb loaded. Nodes={self.graph.number_of_nodes()}, "
              f"Edges={self.graph.number_of_edges()}, "
              f"Time={time.time() - start:.2f} sec")

    ###############################################################
    # Build local STRINGdb graph
    ###############################################################
    def load_string_local(self, links_path, info_path, min_confidence_score):
        # 1. Load protein → gene name mapping
        df_info = pd.read_csv(
            info_path,
            sep="\t",
            usecols=["#string_protein_id", "preferred_name"]
        )
        df_info.rename(
            columns={"#string_protein_id": "protein", "preferred_name": "gene"},
            inplace=True
        )

        # 2. Load interaction file
        df_links = pd.read_csv(
            links_path,
            sep=" ",
            usecols=["protein1", "protein2", "combined_score"]
        )

        # 3. Filter early
        df_links = df_links[df_links["combined_score"] >= min_confidence_score]

        # 4. Normalize score to [0, 1]
        df_links["score"] = df_links["combined_score"] / 1000.0
        df_links = df_links[df_links["score"] > 0]  # avoid log(0)

        # 5. Transform to negative log for max-confidence path
        df_links["weight"] = -np.log(df_links["score"])

        # 6. Map STRING protein IDs → gene names
        df_links = (
            df_links
            .merge(df_info, left_on="protein1", right_on="protein", how="left")
            .rename(columns={"gene": "geneA"})
            .drop(columns=["protein"])
            .merge(df_info, left_on="protein2", right_on="protein", how="left")
            .rename(columns={"gene": "geneB"})
            .drop(columns=["protein"])
        )

        # Drop interactions missing a gene name mapping
        df_links = df_links.dropna(subset=["geneA", "geneB"])

        # Select required columns
        df_edges = df_links[["geneA", "geneB", "weight"]]

        # Build NetworkX graph
        G = nx.from_pandas_edgelist(
            df_edges,
            source="geneA",
            target="geneB",
            edge_attr="weight",
            create_using=nx.Graph()
        )

        return G

    ###############################################################
    # Compute MAX-confidence STRINGdb score
    ###############################################################
    def stringdb_score(self, knockout_gene: str, target_genes: list):
        G = self.graph

        # AARS mapping
        if knockout_gene == "AARS":
            knockout_gene = "AARS1"

        target_genes = ["AARS1" if g == "AARS" else g for g in target_genes]

        # If knockout gene not in graph → return -inf for all
        if knockout_gene not in G:
            return [(gene, float("-inf")) for gene in target_genes]

        results = []

        for target in target_genes:
            if target not in G:
                results.append((target, float("-inf")))
                continue

            try:
                # Dijkstra on -log(score) → finds max-confidence path
                neglog_sum = nx.dijkstra_path_length(
                    G, source=knockout_gene, target=target, weight="weight"
                )
                total_conf = np.exp(-neglog_sum)

            except nx.NetworkXNoPath:
                total_conf = float("-inf")

            results.append((target, round(total_conf, 6)))

        # Sort by descending confidence
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    ###############################################################
    # Compute DIRECT STRINGdb score (no Dijkstra, only direct edges)
    ###############################################################
    def stringdb_score_direct(self, knockout_gene: str, target_genes: list):
        """
        Compute direct edge score between knockout_gene and target_genes.
        This is much faster than stringdb_score as it doesn't use Dijkstra.
        Only returns scores for genes with direct edges to knockout_gene.
        
        Args:
            knockout_gene: Source gene
            target_genes: List of target genes to check
            
        Returns:
            List of (gene, score) tuples, sorted by descending score
        """
        G = self.graph

        # AARS mapping
        if knockout_gene == "AARS":
            knockout_gene = "AARS1"

        target_genes = ["AARS1" if g == "AARS" else g for g in target_genes]

        # If knockout gene not in graph → return 0.0 for all
        if knockout_gene not in G:
            return [(gene, 0.0) for gene in target_genes]

        results = []

        for target in target_genes:
            if target not in G:
                results.append((target, 0.0))
                continue

            # Check if there's a direct edge between knockout_gene and target
            if G.has_edge(knockout_gene, target):
                # Get the edge weight (negative log of score)
                weight = G[knockout_gene][target].get("weight", float("inf"))
                # Convert back to score: score = exp(-weight)
                score = np.exp(-weight)
                results.append((target, round(score, 6)))
            else:
                # No direct edge
                results.append((target, 0.0))

        # Sort by descending score
        results.sort(key=lambda x: x[1], reverse=True)
        return results