
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ------------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ------------------------------------------------------------

df = pd.read_csv("movies_cleaned.csv")

print("=" * 60)
print("MOVIEIQ - PHASE 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"Dataset Shape: {df.shape}")


# ------------------------------------------------------------
# 2. CREATE ASSETS FOLDER
# ------------------------------------------------------------

os.makedirs("assets", exist_ok=True)

print("\nAssets folder ready.")


# ============================================================
# EDA 1 — BUDGET VS REVENUE
# ============================================================

print("\n" + "=" * 60)
print("EDA 1 — BUDGET VS REVENUE")
print("=" * 60)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="budget",
    y="revenue",
    hue="success",
    alpha=0.7
)

plt.title("Budget vs Revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.legend(
    title="Success",
    labels=["Not Successful", "Successful"]
)

plt.tight_layout()

plt.savefig(
    "assets/01_budget_vs_revenue.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Calculate correlation
budget_revenue_corr = df["budget"].corr(df["revenue"])

print(f"Budget-Revenue Correlation: {budget_revenue_corr:.3f}")


# ============================================================
# EDA 2 — GENRE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("EDA 2 — GENRE DISTRIBUTION")
print("=" * 60)

genre_counts = df["genre"].value_counts()

print("\nMovies by Genre:")
print(genre_counts)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=genre_counts.values,
    y=genre_counts.index
)

plt.title("Number of Movies by Genre")
plt.xlabel("Number of Movies")
plt.ylabel("Genre")

plt.tight_layout()

plt.savefig(
    "assets/02_genre_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# EDA 3 — GENRE SUCCESS RATE
# ============================================================

print("\n" + "=" * 60)
print("EDA 3 — GENRE SUCCESS RATE")
print("=" * 60)

genre_success = (
    df.groupby("genre")["success"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print("\nSuccess Rate by Genre:")
print(genre_success.round(2))


plt.figure(figsize=(10, 6))

sns.barplot(
    x=genre_success.values,
    y=genre_success.index
)

plt.title("Movie Success Rate by Genre")
plt.xlabel("Success Rate (%)")
plt.ylabel("Genre")

plt.xlim(0, 100)

plt.tight_layout()

plt.savefig(
    "assets/03_genre_success_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# EDA 4 — POPULARITY VS SUCCESS
# ============================================================

print("\n" + "=" * 60)
print("EDA 4 — POPULARITY VS SUCCESS")
print("=" * 60)

popularity_summary = (
    df.groupby("success")["popularity"]
    .agg(["mean", "median", "min", "max"])
)

print("\nPopularity Summary:")
print(popularity_summary.round(3))


plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="success",
    y="popularity"
)

plt.title("Popularity Distribution by Movie Success")
plt.xlabel("Success (0 = Not Successful, 1 = Successful)")
plt.ylabel("Popularity")

plt.tight_layout()

plt.savefig(
    "assets/04_popularity_vs_success.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# EDA 5 — RUNTIME VS SUCCESS
# ============================================================

print("\n" + "=" * 60)
print("EDA 5 — RUNTIME VS SUCCESS")
print("=" * 60)

runtime_summary = (
    df.groupby("success")["runtime"]
    .agg(["mean", "median", "min", "max"])
)

print("\nRuntime Summary:")
print(runtime_summary.round(3))


plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="success",
    y="runtime"
)

plt.title("Runtime Distribution by Movie Success")
plt.xlabel("Success (0 = Not Successful, 1 = Successful)")
plt.ylabel("Runtime (Minutes)")

plt.tight_layout()

plt.savefig(
    "assets/05_runtime_vs_success.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# EDA 6 — VOTE AVERAGE VS SUCCESS
# ============================================================

print("\n" + "=" * 60)
print("EDA 6 — VOTE AVERAGE VS SUCCESS")
print("=" * 60)

vote_summary = (
    df.groupby("success")["vote_average"]
    .agg(["mean", "median", "min", "max"])
)

print("\nVote Average Summary:")
print(vote_summary.round(3))


plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="success",
    y="vote_average"
)

plt.title("Vote Average Distribution by Movie Success")
plt.xlabel("Success (0 = Not Successful, 1 = Successful)")
plt.ylabel("Vote Average")

plt.tight_layout()

plt.savefig(
    "assets/06_vote_average_vs_success.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# EDA 7 — CORRELATION HEATMAP
# ============================================================

print("\n" + "=" * 60)
print("EDA 7 — CORRELATION HEATMAP")
print("=" * 60)

numeric_columns = [
    "budget",
    "revenue",
    "popularity",
    "runtime",
    "vote_average"
]

correlation_matrix = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))


plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")

plt.tight_layout()

plt.savefig(
    "assets/07_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 8. FINAL EDA SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 — EDA SUMMARY")
print("=" * 60)

print(f"\nBudget-Revenue Correlation: {budget_revenue_corr:.3f}")

print("\nAverage values by Success:")
print(
    df.groupby("success")[
        ["budget", "revenue", "popularity", "runtime", "vote_average"]
    ].mean().round(3)
)

print("\nHighest Success Rate Genre:")
print(
    genre_success.index[0],
    f"({genre_success.iloc[0]:.2f}%)"
)

print("\nLowest Success Rate Genre:")
print(
    genre_success.index[-1],
    f"({genre_success.iloc[-1]:.2f}%)"
)


# ============================================================
# 9. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("7 EDA charts have been saved inside the 'assets' folder.")
