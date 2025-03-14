import pandas as pd
import networkx as nx
import community  # Python-Louvain for clustering
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network  # For interactive visualization
import community.community_louvain as community  # Correct import for Louvain clustering


#  Step 1: Setup Logging 
log_filename = "network_clusters.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

#  Step 2: Command-Line Arguments 
parser = argparse.ArgumentParser(description="Cluster compounds based on similarity and save to a file.")

parser.add_argument("--input", 
                    type=str, 
                    default="pairwise_compound_distances.csv",
                    help="Input file containing the pairwise compound distance matrix")

parser.add_argument("--output", 
                    type=str, 
                    default="compound_clusters.tsv",
                    help="Output file for cluster assignments")

parser.add_argument("--similarity", 
                    type=float, 
                    default=0.5,
                    help="Similarity threshold for clustering")

parser.add_argument("--mcp-only", 
                    action="store_true",
                    help="Filter to only include MCP compounds")

args = parser.parse_args()

#  Step 3: Load Distance Matrix 
logging.info(f"Loading distance matrix from {args.input}")
logging.info("... this takes a long time ... go and have a beer")
dist_df = pd.read_csv(args.input, index_col=0)

filter_mcp_only = args.mcp_only
distance_threshold = args.similarity

#  Step 4: Efficiently Filter for Close Compounds (Vectorized)
logging.info(f"Filtering large dataset efficiently...")

# Convert distance DataFrame into long-form table for efficient filtering
dist_long_df = dist_df.stack().reset_index()
dist_long_df.columns = ["Compound1", "Compound2", "Distance"]

# Remove self-comparisons (Compound1 == Compound2)
dist_long_df = dist_long_df[dist_long_df["Compound1"] != dist_long_df["Compound2"]]

# Apply distance threshold
dist_long_df = dist_long_df[dist_long_df["Distance"] < distance_threshold]

# Apply MCP filtering if required
if filter_mcp_only:
    dist_long_df = dist_long_df[
        (dist_long_df["Compound1"].str.startswith("MCP")) | 
        (dist_long_df["Compound2"].str.startswith("MCP"))
    ]

# Convert filtered data back into a list of edges
edges = list(dist_long_df.itertuples(index=False, name=None))

logging.info(f"Finished filtering: {len(edges)} compound connections found.")

#  Step 5: Create Network 
G = nx.Graph()
for compound1, compound2, distance in edges:
    G.add_edge(compound1, compound2, weight=distance)

logging.info("Network graph created.")

#  Step 6: Detect Clusters (Communities) 
partition = community.best_partition(G, weight="weight")

for node, cluster in partition.items():
    G.nodes[node]["cluster"] = cluster

num_clusters = len(set(partition.values()))
logging.info(f"Detected {num_clusters} clusters.")

#  Step 7: Save Cluster Information 
cluster_df = pd.DataFrame(list(partition.items()), columns=["Compound", "Cluster"])

# Compute cluster sizes
cluster_size_df = cluster_df["Cluster"].value_counts().reset_index()
cluster_size_df.columns = ["Cluster", "Size"]
cluster_size_df.to_csv("cluster_summary.tsv", sep="\t", index=False)

logging.info("Cluster summary saved to 'cluster_summary.tsv'.")

# Compute average similarity within each cluster
cluster_similarities = []
for compound, cluster in partition.items():
    cluster_edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if u == compound or v == compound]
    avg_similarity = sum(d for _, _, d in cluster_edges) / len(cluster_edges) if cluster_edges else None
    cluster_similarities.append(avg_similarity)

cluster_df["Avg_Similarity"] = cluster_similarities
cluster_df.to_csv(args.output, index=False, sep="\t")

logging.info(f"Cluster assignments saved to {args.output}")

#  Step 8: Identify Strongest & Weakest Connections 
edges_sorted = sorted(edges, key=lambda x: x[2])
strongest_connections = edges_sorted[:10]
weakest_connections = edges_sorted[-10:]

strongest_weakest_df = pd.DataFrame(strongest_connections + weakest_connections,
                                    columns=["Compound1", "Compound2", "Distance"])
strongest_weakest_df.to_csv("strongest_weakest_connections.tsv", sep="\t", index=False)

logging.info("Strongest & weakest connections saved to 'strongest_weakest_connections.tsv'.")

#  Step 9: Visualize Compound Distance Distribution 
plt.figure(figsize=(8, 6))
sns.histplot([dist for _, _, dist in edges], bins=30, kde=True, color="blue")

plt.xlabel("Compound Distance")
plt.ylabel("Frequency")
plt.title("Distribution of Compound Pairwise Distances")
plt.savefig("compound_distance_distribution.pdf", format="pdf", bbox_inches="tight")
plt.close()

logging.info("Distance distribution saved to 'compound_distance_distribution.pdf'.")

#  Step 10: Scatterplot of Similarity vs. Cluster Size 
plt.figure(figsize=(8, 6))
sns.scatterplot(x=cluster_size_df["Size"], y=cluster_df["Avg_Similarity"], alpha=0.7, color="blue")

plt.xlabel("Cluster Size")
plt.ylabel("Average Similarity")
plt.title("Cluster Size vs. Similarity")
plt.savefig("cluster_similarity_vs_size.pdf", format="pdf", bbox_inches="tight")
plt.close()

logging.info("Scatterplot of similarity vs. cluster size saved to 'cluster_similarity_vs_size.pdf'.")

#  Step 11: Create Interactive Network Visualization 
net = Network(height="750px", width="100%", notebook=False)

for node in G.nodes:
    cluster_id = partition[node]
    net.add_node(str(node), label=str(node), title=f"Cluster: {cluster_id}", group=cluster_id)

for compound1, compound2, distance in edges:
    net.add_edge(str(compound1), str(compound2), title=f"Distance: {distance:.3f}")


net.save_graph("compound_network.html")


logging.info("Interactive network visualization saved to 'compound_network.html'.")
logging.info("Processing complete!")
