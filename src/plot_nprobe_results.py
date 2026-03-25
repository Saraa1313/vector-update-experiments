import pandas as pd
import matplotlib.pyplot as plt
import os

# Load your results
df = pd.read_csv("results/nprobe_minio.csv")

os.makedirs("graphs", exist_ok=True)

# 1. Recall vs nprobe
plt.figure()
plt.plot(df["nprobe"], df["recall_at_10"], marker="o")
plt.xlabel("nprobe")
plt.ylabel("Recall@10")
plt.title("Recall vs nprobe")
plt.grid()
plt.savefig("graphs/recall_vs_nprobe.png")
plt.close()

# 2. Latency vs nprobe
plt.figure()
plt.plot(df["nprobe"], df["avg_latency_per_query_ms"], marker="o", label="Total latency")
plt.plot(df["nprobe"], df["avg_fetch_time_per_query_ms"], marker="s", label="Fetch latency")
plt.xlabel("nprobe")
plt.ylabel("Latency (ms)")
plt.title("Latency vs nprobe")
plt.legend()
plt.grid()
plt.savefig("graphs/latency_vs_nprobe.png")
plt.close()

# 3. Bytes vs nprobe
plt.figure()
plt.plot(df["nprobe"], df["avg_bytes_per_query"] / (1024*1024), marker="o")
plt.xlabel("nprobe")
plt.ylabel("MB per query")
plt.title("Data fetched vs nprobe")
plt.grid()
plt.savefig("graphs/bytes_vs_nprobe.png")
plt.close()

# 4. Recall vs latency tradeoff
plt.figure()
plt.plot(df["avg_latency_per_query_ms"], df["recall_at_10"], marker="o")
for _, row in df.iterrows():
    plt.annotate(int(row["nprobe"]),
                 (row["avg_latency_per_query_ms"], row["recall_at_10"]))
plt.xlabel("Latency (ms)")
plt.ylabel("Recall@10")
plt.title("Recall vs Latency Tradeoff")
plt.grid()
plt.savefig("graphs/recall_latency_tradeoff.png")
plt.close()

print("All graphs saved in graphs/ folder")
