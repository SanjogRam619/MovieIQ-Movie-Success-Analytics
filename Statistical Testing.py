# ============================================================
# MOVIEIQ — PHASE 3: STATISTICAL TESTING
# ============================================================

import pandas as pd
import numpy as np
import ast

from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency


# ------------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ------------------------------------------------------------

df = pd.read_csv("movies_cleaned.csv")

print("=" * 60)
print("MOVIEIQ — PHASE 3: STATISTICAL TESTING")
print("=" * 60)

print(f"Dataset Shape: {df.shape}")


# ------------------------------------------------------------
# 2. SIGNIFICANCE LEVEL
# ------------------------------------------------------------

alpha = 0.05

print("\nSignificance Level (α):", alpha)


# ============================================================
# T-TEST 1 — POPULARITY
# ============================================================

print("\n" + "=" * 60)
print("T-TEST 1 — POPULARITY")
print("=" * 60)

successful_popularity = df.loc[
    df["success"] == 1,
    "popularity"
]

unsuccessful_popularity = df.loc[
    df["success"] == 0,
    "popularity"
]


t_stat, p_value = ttest_ind(
    successful_popularity,
    unsuccessful_popularity,
    equal_var=False
)


print("\nMean Popularity:")
print(
    f"Successful Movies   : "
    f"{successful_popularity.mean():.3f}"
)

print(
    f"Unsuccessful Movies : "
    f"{unsuccessful_popularity.mean():.3f}"
)

print(f"\nT-Statistic : {t_stat:.4f}")
print(f"P-Value     : {p_value:.4f}")


if p_value < alpha:

    print("\nResult: STATISTICALLY SIGNIFICANT")
    print("Reject the Null Hypothesis (H₀).")
    print(
        "Popularity differs significantly between "
        "successful and unsuccessful movies."
    )

else:

    print("\nResult: NOT STATISTICALLY SIGNIFICANT")
    print("Fail to Reject the Null Hypothesis (H₀).")
    print(
        "There is not enough evidence to conclude that "
        "popularity differs significantly."
    )


# ============================================================
# T-TEST 2 — RUNTIME
# ============================================================

print("\n" + "=" * 60)
print("T-TEST 2 — RUNTIME")
print("=" * 60)

successful_runtime = df.loc[
    df["success"] == 1,
    "runtime"
]

unsuccessful_runtime = df.loc[
    df["success"] == 0,
    "runtime"
]


t_stat_runtime, p_value_runtime = ttest_ind(
    successful_runtime,
    unsuccessful_runtime,
    equal_var=False
)


print("\nMean Runtime:")
print(
    f"Successful Movies   : "
    f"{successful_runtime.mean():.3f} minutes"
)

print(
    f"Unsuccessful Movies : "
    f"{unsuccessful_runtime.mean():.3f} minutes"
)

print(f"\nT-Statistic : {t_stat_runtime:.4f}")
print(f"P-Value     : {p_value_runtime:.4f}")


if p_value_runtime < alpha:

    print("\nResult: STATISTICALLY SIGNIFICANT")
    print("Reject the Null Hypothesis (H₀).")

else:

    print("\nResult: NOT STATISTICALLY SIGNIFICANT")
    print("Fail to Reject the Null Hypothesis (H₀).")


# ============================================================
# T-TEST 3 — VOTE AVERAGE
# ============================================================

print("\n" + "=" * 60)
print("T-TEST 3 — VOTE AVERAGE")
print("=" * 60)

successful_votes = df.loc[
    df["success"] == 1,
    "vote_average"
]

unsuccessful_votes = df.loc[
    df["success"] == 0,
    "vote_average"
]


t_stat_vote, p_value_vote = ttest_ind(
    successful_votes,
    unsuccessful_votes,
    equal_var=False
)


print("\nMean Vote Average:")
print(
    f"Successful Movies   : "
    f"{successful_votes.mean():.3f}"
)

print(
    f"Unsuccessful Movies : "
    f"{unsuccessful_votes.mean():.3f}"
)

print(f"\nT-Statistic : {t_stat_vote:.4f}")
print(f"P-Value     : {p_value_vote:.4f}")


if p_value_vote < alpha:

    print("\nResult: STATISTICALLY SIGNIFICANT")
    print("Reject the Null Hypothesis (H₀).")

else:

    print("\nResult: NOT STATISTICALLY SIGNIFICANT")
    print("Fail to Reject the Null Hypothesis (H₀).")


# ============================================================
# CHI-SQUARE TEST — GENRE VS SUCCESS
# ============================================================

print("\n" + "=" * 60)
print("CHI-SQUARE TEST — GENRE VS SUCCESS")
print("=" * 60)


# Create contingency table

contingency_table = pd.crosstab(
    df["genre"],
    df["success"]
)


print("\nContingency Table:")
print(contingency_table)


# Perform Chi-Square test

chi2_stat, chi2_p_value, degrees_freedom, expected = (
    chi2_contingency(contingency_table)
)


print("\nChi-Square Statistic :", round(chi2_stat, 4))
print("Degrees of Freedom   :", degrees_freedom)
print("P-Value              :", round(chi2_p_value, 4))


if chi2_p_value < alpha:

    print("\nResult: STATISTICALLY SIGNIFICANT")
    print("Reject the Null Hypothesis (H₀).")
    print(
        "Genre and movie success are significantly associated."
    )

else:

    print("\nResult: NOT STATISTICALLY SIGNIFICANT")
    print("Fail to Reject the Null Hypothesis (H₀).")
    print(
        "There is not enough evidence to conclude that "
        "genre is associated with movie success."
    )


# ============================================================
# FINAL STATISTICAL TEST SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PHASE 3 — STATISTICAL TEST SUMMARY")
print("=" * 60)

print("\nT-Test Results:")

print(
    f"Popularity    → p-value = {p_value:.4f}"
)

print(
    f"Runtime       → p-value = {p_value_runtime:.4f}"
)

print(
    f"Vote Average  → p-value = {p_value_vote:.4f}"
)

print("\nChi-Square Test:")

print(
    f"Genre vs Success → p-value = "
    f"{chi2_p_value:.4f}"
)

print("\nSignificance Level:", alpha)

print("\n" + "=" * 60)
print("PHASE 3 COMPLETED SUCCESSFULLY")
print("=" * 60)
