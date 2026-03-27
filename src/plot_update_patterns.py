import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("results/update_matrix.csv")

# -------------------------------
# 1) Recall vs Mutation Fraction (Low vs High Drift)
# -------------------------------
plt.figure(figsize=(7, 5))

for drift in sorted(df["drift_mode"].unique()):
    subset = df[df["drift_mode"] == drift]
    avg = subset.groupby("mutation_fraction")["recall"].mean().sort_index()
    plt.plot(avg.index, avg.values, marker="o", label=drift)

plt.xlabel("Mutation Fraction")
plt.ylabel("Recall")
plt.title("Update Pattern: Low vs High Drift")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("update_pattern_drift_vs_fraction.png", dpi=300)
plt.show()


# -------------------------------
# 2) Range vs Cluster Subset Impact
# -------------------------------
plt.figure(figsize=(7, 5))

for update_type in sorted(df["update_type"].unique()):
    subset = df[df["update_type"] == update_type]
    avg = subset.groupby("mutation_fraction")["recall"].mean().sort_index()
    plt.plot(avg.index, avg.values, marker="o", label=update_type)

plt.xlabel("Mutation Fraction")
plt.ylabel("Recall")
plt.title("Update Pattern: Range vs Cluster Subset")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("update_pattern_range_vs_cluster_subset.png", dpi=300)
plt.show()


# -------------------------------
# 3) Recall vs Query Impact Bucket
# -------------------------------
plt.figure(figsize=(7, 5))

avg = df.groupby("target_bucket")["recall"].mean().sort_index()
plt.plot(avg.index, avg.values, marker="o")

plt.xlabel("Target Bucket (Fraction of Mutated Neighbors)")
plt.ylabel("Recall")
plt.title("Update Pattern: Recall vs Query Impact Bucket")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("update_pattern_query_impact_bucket.png", dpi=300)
plt.show()
