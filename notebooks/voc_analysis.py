import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open("../outputs/voc_scored.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print("Loaded", len(df), "examples.")

df.head()

total = len(df)
cot_helpful = (df['voc'] > 0).sum()
direct_better = total - cot_helpful

print(f"Total examples: {total}")
print(f"CoT helpful (VOC > 0): {cot_helpful} ({cot_helpful / total:.1%})")
print(f"Direct better (VOC <= 0): {direct_better} ({direct_better / total:.1%})")
print("\nAverage VOC:", round(df["voc"].mean(), 3))
print("Average utility:", round(df["utility"].mean(), 3))
print("Average cost:", round(df["cost"].mean(), 3))

plt.figure(figsize=(8, 5))
sns.histplot(df["voc"], bins=15, kde=True, color="darkorange")
plt.axvline(0, linestyle="--", color="gray")
plt.title("Distribution of VOC Scores")
plt.xlabel("Value of Computation (VOC)")
plt.ylabel("Number of Questions")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="cost", y="utility", hue="strategy", palette="Set2")
plt.title("Cost vs Utility of Reasoning Chains")
plt.xlabel("Reasoning Cost (γ × tokens)")
plt.ylabel("Utility (log-prob boost)")
plt.grid(True)
plt.show()

plt.figure(figsize=(4, 4))
df["strategy"].value_counts().plot.pie(
    autopct="%1.0f%%", startangle=90, colors=["#66c2a5", "#fc8d62"]
)
plt.title("Reasoning Strategy Chosen (Based on VOC)")
plt.ylabel("")
plt.show()
df_sorted = df.sort_values("voc", ascending=False)
print("🔼 Top 3 cases where reasoning helped:")
df_sorted.head(3)[["question", "direct", "cot", "voc"]]
print("🔽 Bottom 3 cases where reasoning hurt:")
df_sorted.tail(3)[["question", "direct", "cot", "voc"]]

