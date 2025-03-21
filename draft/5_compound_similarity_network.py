"""
Compound Similarity Network Analysis
------------------------------------

This script processes a pairwise compound distance matrix to identify 
clusters of similar compounds, focusing on MCP* compounds if specified. 
It performs network-based clustering and generates various reports 
and visualizations for downstream analysis.

Workflow:
---------
1. **Load Distance Matrix**
   - Reads the compound pairwise distance matrix from a CSV file.
   - Allows filtering for only MCP* compounds if specified.

2. **Filter for Close Compounds**
   - Converts the distance matrix into a long-form table.
   - Removes self-comparisons (compounds vs. themselves).
   - Filters pairs based on a user-defined similarity threshold.
   - If MCP filtering is enabled, only includes edges where at least 
     one compound starts with 'MCP'.

3. **Create a Network Graph**
   - Constructs a NetworkX graph where:
     - Nodes represent compounds.
     - Edges represent compound pairs with similarity below the threshold.
   - Optionally removes non-MCP nodes to keep the network MCP-only.

4. **Detect Clusters Using Louvain Community Detection**
   - Assigns each compound to a cluster based on network topology.
   - Reports the total number of detected clusters.

5. **Save Cluster Assignments**
   - Saves a tab-separated file (`compound_clusters.tsv`) listing:
     - Each compound.
     - Its assigned cluster.
     - Its average similarity within the cluster.

6. **Identify Strongest & Weakest Connections**
   - Extracts and saves:
     - The **10 strongest** (most similar) compound connections.
     - The **10 weakest** (least similar but still within the threshold).
   - Results saved to `strongest_weakest_connections.tsv`.

7. **Generate Visualizations**
   - **Distance Distribution Histogram (`compound_distance_distribution.pdf`)**
     - Shows how compound pairwise distances are distributed.
   - **Cluster Size vs. Similarity Scatterplot (`cluster_similarity_vs_size.pdf`)**
     - Plots the relationship between cluster size and average similarity.
   - **Interactive Network Graph (`compound_network.html`)**
     - Displays a zoomable, clickable network of compounds and their relationships.

8. **Save a Cluster Summary**
   - Outputs `cluster_summary.tsv` with:
     - The total number of compounds in each cluster.
     - The size distribution of clusters.

9. **Final Logging & Completion**
   - Logs all key steps and outputs to `network_clusters.log`.
   - Ensures all outputs are saved without opening files automatically.

Reports Generated:
------------------
1. **Cluster Assignments:** `compound_clusters.tsv`
2. **Cluster Summary:** `cluster_summary.tsv`
3. **Strongest & Weakest Connections:** `strongest_weakest_connections.tsv`
4. **Distance Distribution Histogram:** `compound_distance_distribution.pdf`
5. **Cluster Similarity vs. Size Scatterplot:** `cluster_similarity_vs_size.pdf`
6. **Interactive Network Graph:** `compound_network.html`
7. **Log File:** `network_clusters.log`

Command-Line Options:
---------------------
- `--input`: Path to the input pairwise distance matrix (default: `pairwise_compound_distances.csv`).
- `--output`: Path to save the cluster assignments (default: `compound_clusters.tsv`).
- `--similarity`: Similarity threshold for clustering (default: `0.5`).
- `--mcp-only`: If set, restricts the network to only MCP* compounds.

Usage Example:
--------------

python 5_compound_similarity_network.py --similarity 0.8 -c MCP

"""

import os
import pandas as pd
import networkx as nx
import community  # Python-Louvain for clustering
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from pyvis.network import Network  # For interactive visualization
import community.community_louvain as community  # Correct import for Louvain clustering



import pandas as pd
import logging




#  Step 1: Setup Logging 
log_folder = "similarity_network"
os.makedirs(log_folder, exist_ok=True)  # Ensure output folder exists
log_filename = os.path.join(log_folder, "network_clusters.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
###########################################
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

parser.add_argument("--compound-prefix", 
                    type=str, 
                    default="MCP",
                    help="Specify the prefix of compounds to focus on (default: MCP)")

parser.add_argument("--no-prefix-filtering", 
                    action="store_true",
                    help="Include all compounds, ignoring prefix filtering.")

args = parser.parse_args()
###########################################


# Create formatted output filename prefix
prefix_label = "ALL" if args.no_prefix_filtering else args.compound_prefix
similarity_label = f"{args.similarity:.2f}"
output_prefix = os.path.join(log_folder, f"{prefix_label}_sim{similarity_label}")

#  Step 3: Load Distance Matrix 
logging.info(f"Loading distance matrix from {args.input}")
logging.info("... this takes a long time ... go and have a beer")
logging.info(f"Loading and cleaning pairwise matrix from {args.input}")
dist_df = pd.read_csv(args.input, index_col=0)
logging.info(f"Parsed distance matrix shape: {dist_df.shape}")



# Step 3.2: Clustered heatmap with MCP* labels
logging.info("Generating heatmap with similarity clustering and MCP* compound highlights...")

# Ensure the matrix is square and numeric
assert dist_df.shape[0] == dist_df.shape[1], "Distance matrix must be square"
assert (dist_df.columns == dist_df.index).all(), "Matrix rows and columns must match"

# Perform hierarchical clustering
linkage_matrix = linkage(squareform(dist_df.values), method="average")

# Create a clustermap (not just heatmap – includes dendrogram)
compound_labels = dist_df.index.tolist()
is_mcp = [label.startswith("MCP") for label in compound_labels]

# Row/column colours to highlight MCP* rows
row_colours = ["red" if flag else "grey" for flag in is_mcp]

# Create and save clustermap
clustered_path = os.path.join(log_folder, f"{prefix_label}_sim{similarity_label}_clustered_heatmap.pdf")
sns.set(style="white")
cg = sns.clustermap(
    dist_df,
    row_linkage=linkage_matrix,
    col_linkage=linkage_matrix,
    cmap="viridis",
    figsize=(15, 13),
    row_colors=row_colours,
    col_colors=row_colours,
    xticklabels=True,
    yticklabels=True
)

# Set font size for tick labels
cg.ax_heatmap.tick_params(axis='x', labelsize=4)
cg.ax_heatmap.tick_params(axis='y', labelsize=4)

plt.suptitle("Clustered Compound Distance Matrix (MCP* highlighted)", y=1.02)
plt.savefig(clustered_path, bbox_inches="tight")
plt.close()

logging.info(f"Clustered heatmap saved to '{clustered_path}'.")


# SAFEGUARD: Ensure data is aggregated before processing
if "cpd_id" in dist_df.columns and "Library" in dist_df.columns:
    logging.info("Data appears to be at the cell level. Aggregating by compound...")
    dist_df = dist_df.groupby(["cpd_id", "Library"]).mean().reset_index()
    logging.info(f"Data aggregated. New shape: {dist_df.shape}")
else:
    logging.info("Data is already at the compound level. Proceeding with analysis.")

compound_prefix = args.compound_prefix
distance_threshold = args.similarity
disable_prefix_filtering = args.no_prefix_filtering

#  Step 4: Efficiently Filter for Close Compounds (Vectorized)
logging.info(f"Filtering dataset with threshold {distance_threshold}...")

# Convert distance DataFrame into long-form table for efficient filtering
dist_long_df = dist_df.stack().reset_index()
dist_long_df.columns = ["Compound1", "Compound2", "Distance"]

# Ensure Distance is numeric (force errors to NaN)
dist_long_df["Distance"] = pd.to_numeric(dist_long_df["Distance"], errors="coerce")

# Drop rows where distance could not be converted
dist_long_df = dist_long_df.dropna(subset=["Distance"])


# Remove self-comparisons (Compound1 == Compound2)
dist_long_df = dist_long_df[dist_long_df["Compound1"] != dist_long_df["Compound2"]]

# Apply distance threshold
dist_long_df = dist_long_df[dist_long_df["Distance"] < distance_threshold]

# Apply filtering for the specified compound prefix (if not disabled)
if not disable_prefix_filtering:
    dist_long_df = dist_long_df[
        (dist_long_df["Compound1"].str.startswith(compound_prefix)) | 
        (dist_long_df["Compound2"].str.startswith(compound_prefix))
    ]

logging.info(f"Finished filtering: {len(dist_long_df)} compound connections found.")

# Convert filtered data back into a list of edges
edges = list(dist_long_df.itertuples(index=False, name=None))

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
cluster_size_df.to_csv(f"{output_prefix}_cluster_summary.tsv", sep="\t", index=False)

logging.info(f"Cluster summary saved to '{output_prefix}_cluster_summary.tsv'.")

# Compute average similarity within each cluster
cluster_similarities = []
for compound, cluster in partition.items():
    cluster_edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if u == compound or v == compound]
    avg_similarity = sum(d for _, _, d in cluster_edges) / len(cluster_edges) if cluster_edges else None
    cluster_similarities.append(avg_similarity)

cluster_df["Avg_Similarity"] = cluster_similarities
cluster_df.to_csv(f"{output_prefix}_compound_clusters.tsv", index=False, sep="\t")

logging.info(f"Cluster assignments saved to '{output_prefix}_compound_clusters.tsv'.")

#  Step 8: Identify Strongest & Weakest Connections 
edges_sorted = sorted(edges, key=lambda x: x[2])
strongest_connections = edges_sorted[:10]
weakest_connections = edges_sorted[-10:]

strongest_weakest_df = pd.DataFrame(strongest_connections + weakest_connections,
                                    columns=["Compound1", "Compound2", "Distance"])
strongest_weakest_df.to_csv(f"{output_prefix}_strongest_weakest_connections.tsv", sep="\t", index=False)

logging.info(f"Strongest & weakest connections saved to '{output_prefix}_strongest_weakest_connections.tsv'.")

#  Step 9: Visualize Compound Distance Distribution 
plt.figure(figsize=(8, 6))
sns.histplot([dist for _, _, dist in edges], bins=30, kde=True, color="blue")

plt.xlabel("Compound Distance")
plt.ylabel("Frequency")
plt.title("Distribution of Compound Pairwise Distances")
plt.savefig(f"{output_prefix}_distance_distribution.pdf", format="pdf", bbox_inches="tight")
plt.close()

logging.info(f"Distance distribution saved to '{output_prefix}_distance_distribution.pdf'.")

#  Step 10: Scatterplot of Similarity vs. Cluster Size 
plt.figure(figsize=(8, 6))
sns.scatterplot(x=cluster_size_df["Size"], y=cluster_df["Avg_Similarity"], alpha=0.7, color="blue")

plt.xlabel("Cluster Size")
plt.ylabel("Average Similarity")
plt.title("Cluster Size vs. Similarity")
plt.savefig(f"{output_prefix}_cluster_similarity_vs_size.pdf", format="pdf", bbox_inches="tight")
plt.close()

logging.info(f"Scatterplot of similarity vs. cluster size saved to '{output_prefix}_cluster_similarity_vs_size.pdf'.")

#  Step 11: Create Interactive Network Visualization 
net = Network(height="750px", width="100%", notebook=False)

for node in G.nodes:
    cluster_id = partition[node]
    net.add_node(str(node), label=str(node), title=f"Cluster: {cluster_id}", group=cluster_id)

for compound1, compound2, distance in edges:
    net.add_edge(str(compound1), str(compound2), title=f"Distance: {distance:.3f}")

net.save_graph(f"{output_prefix}_network.html")

logging.info(f"Interactive network visualization saved to '{output_prefix}_network.html'.")
logging.info("Processing complete!")