# ============================================================
# MOVIEIQ — PHASE 1: DATA PREPARATION
# ============================================================

import pandas as pd
import numpy as np
import ast


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("movies.csv")

print("=" * 60)
print("MOVIEIQ - DATASET LOADED")
print("=" * 60)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ------------------------------------------------------------
# 2. BASIC DATASET UNDERSTANDING
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

print(df.columns.tolist())


print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()


# ------------------------------------------------------------
# 3. FIRST 5 RECORDS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

print(df.head())


# ------------------------------------------------------------
# 4. SUMMARY STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(df.describe().T)


# ------------------------------------------------------------
# 5. MISSING VALUE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)


# ------------------------------------------------------------
# 6. ZERO VALUE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ZERO VALUE CHECK")
print("=" * 60)

zero_budget = (df["budget"] == 0).sum()
zero_revenue = (df["revenue"] == 0).sum()

print(f"Zero budget records  : {zero_budget}")
print(f"Zero revenue records : {zero_revenue}")


# ------------------------------------------------------------
# 7. DUPLICATE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

duplicate_rows = df.duplicated().sum()
duplicate_titles = df["title"].duplicated().sum()

print(f"Duplicate rows   : {duplicate_rows}")
print(f"Duplicate titles : {duplicate_titles}")


# ------------------------------------------------------------
# 8. GENRE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("GENRE ANALYSIS")
print("=" * 60)

empty_genres = (df["genres"] == "[]").sum()

print(f"Movies with empty genre : {empty_genres}")
print(f"Percentage              : {empty_genres / len(df) * 100:.2f}%")


# ------------------------------------------------------------
# 9. CLEAN / EXTRACT GENRE
# ------------------------------------------------------------

def extract_genre(value):

    try:
        genre_list = ast.literal_eval(value)

        if isinstance(genre_list, list) and len(genre_list) > 0:

            genre_name = genre_list[0].get("name")

            if genre_name:
                return genre_name

        return "Unknown"

    except:
        return "Unknown"


df["genre"] = df["genres"].apply(extract_genre)


print("\nGenre distribution:")
print(df["genre"].value_counts())


# ------------------------------------------------------------
# 10. CREATE TARGET VARIABLE — SUCCESS
# ------------------------------------------------------------

# Success = 1 when revenue > budget
# Success = 0 otherwise

df["success"] = (df["revenue"] > df["budget"]).astype(int)


# ------------------------------------------------------------
# 11. SUCCESS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SUCCESS DISTRIBUTION")
print("=" * 60)

success_counts = df["success"].value_counts().sort_index()

success_percentage = (
    df["success"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

success_summary = pd.DataFrame({
    "Movie Count": success_counts,
    "Percentage": success_percentage.round(2)
})

success_summary.index = [
    "Not Successful (0)",
    "Successful (1)"
]

print(success_summary)


# ------------------------------------------------------------
# 12. CREATE FINAL CLEAN DATASET
# ------------------------------------------------------------

# Remove the original raw genres column.
# Keep the cleaned genre column.

df_clean = df.drop(columns=["genres"])


# ------------------------------------------------------------
# 13. FINAL DATASET CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL CLEAN DATASET")
print("=" * 60)

print(f"Rows    : {df_clean.shape[0]}")
print(f"Columns : {df_clean.shape[1]}")

print("\nFinal columns:")
print(df_clean.columns.tolist())

print("\nMissing values:")
print(df_clean.isnull().sum())

print("\nFirst 5 cleaned records:")
print(df_clean.head())


# ------------------------------------------------------------
# 14. SAVE CLEANED DATASET
# ------------------------------------------------------------

df_clean.to_csv("movies_cleaned.csv", index=False)


# ------------------------------------------------------------
# 15. COMPLETION MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PHASE 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Cleaned dataset saved as: movies_cleaned.csv")
