# ============================================================
# MOVIEIQ — PHASE 4
# STEP 1: FEATURE SELECTION & TARGET PREPARATION
# ============================================================

import pandas as pd


# ------------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ------------------------------------------------------------

df = pd.read_csv("../movies_cleaned.csv")

print("=" * 60)
print("MOVIEIQ — PHASE 4: STEP 1")
print("FEATURE SELECTION & TARGET PREPARATION")
print("=" * 60)

print(f"Dataset Shape: {df.shape}")


# ------------------------------------------------------------
# 2. SELECT FEATURES (X)
# ------------------------------------------------------------

features = [
    "budget",
    "popularity",
    "runtime",
    "vote_average",
    "genre"
]

X = df[features]


# ------------------------------------------------------------
# 3. SELECT TARGET (y)
# ------------------------------------------------------------

y = df["success"]


# ------------------------------------------------------------
# 4. DISPLAY SELECTED FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SELECTED FEATURES (X)")
print("=" * 60)

print(X.head())

print("\nFeatures used:")
print(X.columns.tolist())


# ------------------------------------------------------------
# 5. DISPLAY TARGET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TARGET VARIABLE (y)")
print("=" * 60)

print(y.head())

print("\nTarget name:")
print(y.name)


# ------------------------------------------------------------
# 6. CHECK SHAPES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE & TARGET SHAPES")
print("=" * 60)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ------------------------------------------------------------
# 7. CHECK TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(y.value_counts())

print("\nTarget percentages:")
print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ------------------------------------------------------------
# 8. VERIFY DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE DATA TYPES")
print("=" * 60)

print(X.dtypes)


# ------------------------------------------------------------
# 9. VERIFY EXCLUDED COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EXCLUDED COLUMNS")
print("=" * 60)

print("Excluded: revenue")
print("Reason  : Revenue directly determines the success target.")

print("\nExcluded: title")
print("Reason  : Movie title is an identifier and is not a useful")
print("          numerical predictor for this model.")


# ------------------------------------------------------------
# 10. STEP 1 COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Features selected:")
print(features)

print("\nTarget selected:")
print("success")
