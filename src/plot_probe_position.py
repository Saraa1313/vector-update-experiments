import os
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results/probe_position.csv")

os.makedirs("graphs", exist_ok=True)

# Keep only the latest experiment rows if duplicates exist
# Optional: comment this out if not needed
df = df.sort_values(["case", "index_state"]).reset_index(drop=True)

# Make labels nicer
case_name_map = {
    "closest_probed_mutated": "Closest probed mutated",
    "farthest_probed_mutated": "Farthest probed mutated",
}
df["case_label"] = df["case"].map(case_name_map).fillna(df["case"])

# -----------------------------
# Plot 1: Recall by case and index state
# -----------------------------
pivot_recall = df.pivot(index="case_label", columns="index_state", values="recall")

ax = pivot_recall.plot(kind="bar", figsize=(8, 5))
ax.set_title("Recall under Stale vs Refreshed Assignments")
ax.set_xlabel("Mutation case")
ax.set_ylabel("Recall@10")
ax.set_ylim(0, min(1.05, max(1.0, df["recall"].max() + 0.05)))
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("graphs/probe_position_recall_bar.png", dpi=200)
plt.close()

# -----------------------------
# Plot 2: Recovery from stale to refreshed
# -----------------------------
# refreshed - stale
recovery_rows = []
for case_label in pivot_recall.index:
    stale_val = pivot_recall.loc[case_label, "stale"] if "stale" in pivot_recall.columns else None
    refreshed_val = pivot_recall.loc[case_label, "refreshed"] if "refreshed" in pivot_recall.columns else None
    if stale_val is not None and refreshed_val is not None:
        recovery_rows.append({
            "case_label": case_label,
            "recovery": refreshed_val - stale_val
        })

recovery_df = pd.DataFrame(recovery_rows)

plt.figure(figsize=(8, 5))
plt.bar(recovery_df["case_label"], recovery_df["recovery"])
plt.title("Recall Recovery from Refreshing Assignments")
plt.xlabel("Mutation case")
plt.ylabel("Recall gain (refreshed - stale)")
plt.grid(axis="y", alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("graphs/probe_position_recovery.png", dpi=200)
plt.close()

# -----------------------------
# Plot 3: Recall drop relative to refreshed
# -----------------------------
drop_rows = []
for case_label in pivot_recall.index:
    stale_val = pivot_recall.loc[case_label, "stale"] if "stale" in pivot_recall.columns else None
    refreshed_val = pivot_recall.loc[case_label, "refreshed"] if "refreshed" in pivot_recall.columns else None
    if stale_val is not None and refreshed_val is not None:
        drop_rows.append({
            "case_label": case_label,
            "stale_drop_from_refreshed": refreshed_val - stale_val
        })

drop_df = pd.DataFrame(drop_rows)

plt.figure(figsize=(8, 5))
plt.bar(drop_df["case_label"], drop_df["stale_drop_from_refreshed"])
plt.title("Recall Loss Due to Stale Partition Placement")
plt.xlabel("Mutation case")
plt.ylabel("Recall drop vs refreshed")
plt.grid(axis="y", alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("graphs/probe_position_stale_drop.png", dpi=200)
plt.close()

print("Saved plots:")
print("graphs/probe_position_recall_bar.png")
print("graphs/probe_position_recovery.png")
print("graphs/probe_position_stale_drop.png")
