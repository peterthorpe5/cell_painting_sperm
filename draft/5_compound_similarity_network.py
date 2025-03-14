import pandas as pd
import networkx as nx
import community  # Python-Louvain for clustering

# Step 1: Load Distance Matrix ----------------
dist_df = pd.read_csv("pairwise_compound_distances.csv", index_col=0)

# User Setting: Filter Only MCP Compounds ----------------
filter_mcp_only = True  # Set to False if you want to include all compounds

# Step 2: Filter for Close Compounds ----------------
distance_threshold = 1.5  # Adjust this for "closeness"

edges = []
for compound1 in dist_df.index:
    for compound2 in dist_df.columns:
        if compound1 != compound2:  # Exclude self-comparisons
            distance = dist_df.loc[compound1, compound2]

            # Ensure distance is a single numeric value
            if isinstance(distance, pd.Series):
                distance = distance.iloc[0] if not distance.empty else float("inf")  # Assign infinity if empty

            # Apply filtering
            if pd.notna(distance) and distance < distance_threshold:
                if not filter_mcp_only or (compound1.startswith("MCP") or compound2.startswith("MCP")):
                    edges.append((compound1, compound2, distance))


# Step 3: Create Network ----------------
G = nx.Graph()

for compound1, compound2, distance in edges:
    G.add_edge(compound1, compound2, weight=distance)

# Step 4: Detect Clusters (Communities) ----------------
partition = community.best_partition(G, weight="weight")  # Louvain clustering

# Assign cluster labels to nodes
for node, cluster in partition.items():
    G.nodes[node]["cluster"] = cluster

# Step 5: Save Cluster Information to CSV ----------------
# Convert partition dictionary to a DataFrame
cluster_df = pd.DataFrame(list(partition.items()), columns=["Compound", "Cluster"])

# Calculate average similarity within each cluster
cluster_similarities = []
for compound, cluster in partition.items():
    # Get all edges where this compound is involved
    cluster_edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if u == compound or v == compound]
    
    # Compute the average similarity within the cluster
    avg_similarity = sum(d for _, _, d in cluster_edges) / len(cluster_edges) if cluster_edges else None
    cluster_similarities.append(avg_similarity)

# Add average similarity to DataFrame
cluster_df["Avg_Similarity"] = cluster_similarities


# Save to a tab-separated file
cluster_df.to_csv("compound_clusters.tsv", index=False, sep="\t")

print("Cluster assignments saved to 'compound_clusters.tsv'")
