import pandas as pd
import matplotlib.pyplot as plt
import os

# Load results
df = pd.read_csv("results/cache_static.csv")

os.makedirs("graphs", exist_ok=True)

# Sort by cache size (important for clean plots)
df = df.sort_values("cache_size_lists")

# =========================
# Figure 1: Latency + Fetch Latency
# =========================
plt.figure()
plt.plot(df["cache_size_lists"], df["avg_latency_per_query_ms"],
         marker="o", label="Total latency")

plt.plot(df["cache_size_lists"], df["avg_fetch_time_per_query_ms"],
         marker="s", linestyle="--", label="Fetch latency")

plt.xlabel("Cache size (number of lists)")
plt.ylabel("Latency (ms)")
plt.title("Latency vs Cache Size")
plt.legend()
plt.grid()

plt.savefig("graphs/cache_latency_combined.png")
plt.close()


# =========================
# Figure 2: Hit Rate + Objects Fetched
# =========================
plt.figure()

plt.plot(df["cache_size_lists"], df["cache_hit_rate"],
         marker="o", label="Cache hit rate")

plt.plot(df["cache_size_lists"], df["avg_objects_fetched_per_query"],
         marker="s", linestyle="--", label="Objects fetched/query")

plt.xlabel("Cache size (number of lists)")
plt.ylabel("Value")
plt.title("Cache Efficiency vs Cache Size")
plt.legend()
plt.grid()

plt.savefig("graphs/cache_hitrate_objects.png")
plt.close()


# =========================
# Figure 3: Cost vs Cache Size
# =========================
plt.figure()

plt.plot(df["cache_size_lists"], df["estimated_total_cost_per_1m_queries"],
         marker="o")

plt.xlabel("Cache size (number of lists)")
plt.ylabel("Cost per 1M queries ($)")
plt.title("Estimated Cost vs Cache Size")
plt.grid()

plt.savefig("graphs/cache_cost.png")
plt.close()


print("Saved all cache plots in graphs/")
