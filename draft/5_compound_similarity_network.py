import pandas as pd
import networkx as nx
import community  # Python-Louvain for clustering
import argparse
import logging

# Step 1: Setup Logging (to File and Console) 
log_filename = "network_clusters.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Write logs to file
        logging.StreamHandler()  # Print logs to terminal
    ]
)

# Step 2: Command-Line Arguments 
parser = argparse.ArgumentParser(description="Cluster compounds based on similarity and save to a file.")

parser.add_argument("--input", type=str, default="pairwise_compound_distances.csv",
                    help="Input file containing the pairwise compound distance matrix (default: pairwise_compound_distances.csv)")

parser.add_argument("--output", type=str, default="compound_clusters.tsv",
                    help="Output file for cluster assignments (default: compound_clusters.tsv)")

parser.add_argument("--similarity", type=float, default=1.5,
                    help="Similarity threshold for clustering (default: 1.5)")

parser.add_argument("--mcp-only", action="store_true",
                    help="Filter to only include MCP compounds (default: False)")

args = parser.parse_args()

# Step 3: Load Distance Matrix 
logging.info(f"Loading distance matrix from {args.input}")
dist_df = pd.read_csv(args.input, index_col=0)

filter_mcp_only = args.mcp_only  # Set to True if flag is used
distance_threshold = args.similarity  # User-defined threshold

# Step 4: Filter for Close Compounds 
logging.info(f"Filtering compounds with similarity threshold {distance_threshold}")

edges = []
for compound1 in dist_df.index:
    for compound2 in dist_df.columns:
        if compound1 != compound2:  # Exclude self-comparisons
            distance = dist_df.loc[compound1, compound2]

            # Ensure distance is a single numeric value
            if isinstance(distance, pd.Series):
                distance = distance.iloc[0] if not distance.empty else float("inf")

            # Apply filtering
            if pd.notna(distance) and distance < distance_threshold:
                if not filter_mcp_only or (compound1.startswith("MCP") or compound2.startswith("MCP")):
                    edges.append((compound1, compound2, distance))

logging.info(f"Filtered {len(edges)} compound connections.")

# Step 5: Create Network 
G = nx.Graph()

for compound1, compound2, distance in edges:
    G.add_edge(compound1, compound2, weight=distance)

logging.info("Network graph created.")

# Step 6: Detect Clusters (Communities) 
partition = community.best_partition(G, weight="weight")  # Louvain clustering

# Assign cluster labels to nodes
for node, cluster in partition.items():
    G.nodes[node]["cluster"] = cluster

num_clusters = len(set(partition.values()))
logging.info(f"Detected {num_clusters} clusters.")

# Step 7: Save Cluster Information 
cluster_df = pd.DataFrame(list(partition.items()), columns=["Compound", "Cluster"])

# Calculate average similarity within each cluster
cluster_similarities = []
for compound, cluster in partition.items():
    cluster_edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if u == compound or v == compound]
    avg_similarity = sum(d for _, _, d in cluster_edges) / len(cluster_edges) if cluster_edges else None
    cluster_similarities.append(avg_similarity)

cluster_df["Avg_Similarity"] = cluster_similarities

# Save to a tab-separated file
cluster_df.to_csv(args.output, index=False, sep="\t")

logging.info(f"Cluster assignments saved to {args.output}")
logging.info("Processing complete!")
