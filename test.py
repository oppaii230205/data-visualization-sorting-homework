import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set style chuyên nghiệp
sns.set_theme(style="whitegrid", context="talk")

# Data (Random Data)
data = {
    "n": [100, 1000, 10000, 100000],
    "Insertion Sort": [1, 100, 10000, 1000000],
    "Quick Sort": [0.2, 2, 30, 400],
    "Counting Sort": [0.1, 1, 10, 100]
}

df = pd.DataFrame(data)

# Convert sang dạng long-form (chuẩn seaborn)
df_melted = df.melt(id_vars="n", 
                    var_name="Algorithm", 
                    value_name="Time (ms)")

# Random Data - Biểu đồ đường (Line Plot) với thang tuyến tính
# plt.figure(figsize=(10, 6))

# sns.lineplot(
#     data=df_melted,
#     x="n",
#     y="Time (ms)",
#     hue="Algorithm",
#     marker="o",
#     linewidth=2.5
# )

# plt.title("Sorting Algorithm Performance (Random Data) - Linear Scale", fontsize=16, weight='bold')
# plt.xlabel("Input Size (n)", fontsize=12)
# plt.ylabel("Execution Time (ms)", fontsize=12)

# plt.legend(title="Algorithm")
# plt.grid(True, linestyle='--', alpha=0.6)

# plt.tight_layout()
# plt.show()

# Random Data - Biểu đồ đường (Line Plot) với thang logarit
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df_melted,
    x="n",
    y="Time (ms)",
    hue="Algorithm",
    marker="o",
    linewidth=2.5
)

plt.yscale("log")  # 🔥 KEY POINT

plt.title("Sorting Algorithm Performance (Random Data) - Log Scale", fontsize=16, weight='bold')
plt.xlabel("Input Size (n)", fontsize=12)
plt.ylabel("Execution Time (ms, log scale)", fontsize=12)

plt.legend(title="Algorithm")
plt.grid(True, which="both", linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()