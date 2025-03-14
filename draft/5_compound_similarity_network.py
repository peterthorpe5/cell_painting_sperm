import pandas as pd
import networkx as nx
import community  # Python-Louvain for clustering
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns

#  Step 1: Setup Logging (to File and Console) 
log_filename = "network_clusters.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Write logs to file
        logging.StreamHandler()  # Print logs to terminal
    ]
)

#  Step 2: Command-Line Arguments 
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

#  Step 3: Load Distance Matrix 
logging.info(f"Loading distance matrix from {args.input}")
dist_df = pd.read_csv(args.input, index_col=0)

filter_mcp_only = args.mcp_only  # Set to True if flag is used
distance_threshold = args.similarity  # User-defined threshold

#  Step 4: Filter for Close Compounds 
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

#  Step 5: Create Network 
G = nx.Graph()

for compound1, compound2, distance in edges:
    G.add_edge(compound1, compound2, weight=distance)

logging.info("Network graph created.")

#  Step 6: Detect Clusters (Communities) 
partition = community.best_partition(G, weight="weight")  # Louvain clustering

# Assign cluster labels to nodes
for node, cluster in partition.items():
    G.nodes[node]["cluster"] = cluster

num_clusters = len(set(partition.values()))
logging.info(f"Detected {num_clusters} clusters.")

#  Step 7: Save Cluster Information 
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

#  Step 8: Identify Strongest & Weakest Connections 
edges_sorted = sorted(edges, key=lambda x: x[2])  # Sort by distance (ascending)

strongest_connections = edges_sorted[:10]  # Smallest distances = Most similar
weakest_connections = edges_sorted[-10:]  # Largest distances but still within threshold

# Save strongest & weakest connections to file
strongest_weakest_df = pd.DataFrame(strongest_connections + weakest_connections, 
                                    columns=["Compound1", "Compound2", "Distance"])
strongest_weakest_df.to_csv("strongest_weakest_connections.tsv", sep="\t", index=False)

logging.info("Top 5 Strongest Connections (Most Similar):")
for c1, c2, dist in strongest_connections[:5]:
    logging.info(f"{c1} ↔ {c2} (Distance: {dist:.4f})")

logging.info("Top 5 Weakest Connections (Least Similar within threshold):")
for c1, c2, dist in weakest_connections[:5]:
    logging.info(f"{c1} ↔ {c2} (Distance: {dist:.4f})")

logging.info("Strongest and weakest connections saved to 'strongest_weakest_connections.tsv'.")

#  Step 9: Visualize Compound Distance Distribution 
plt.figure(figsize=(8, 6))
sns.histplot([dist for _, _, dist in edges], bins=30, kde=True, color="blue")

plt.xlabel("Compound Distance")
plt.ylabel("Frequency")
plt.title("Distribution of Compound Pairwise Distances")
plt.savefig("compound_distance_distribution.pdf", format="pdf", bbox_inches="tight")
plt.close()

logging.info("Compound distance distribution histogram saved to 'compound_distance_distribution.pdf'.")
logging.info("Processing complete!")
