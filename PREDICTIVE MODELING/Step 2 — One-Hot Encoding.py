# ============================================================
# MOVIEIQ — PHASE 4
# STEP 2: ONE-HOT ENCODING
# ============================================================

import pandas as pd

from sklearn.preprocessing import OneHotEncoder


# ------------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ------------------------------------------------------------

df = pd.read_csv("../movies_cleaned.csv")

print("=" * 60)
print("MOVIEIQ — PHASE 4: STEP 2")
print("ONE-HOT ENCODING")
print("=" * 60)

print(f"Original Dataset Shape: {df.shape}")


# ------------------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# ------------------------------------------------------------

features = [
    "budget",
    "popularity",
    "runtime",
    "vote_average",
    "genre"
]

X = df[features]
y = df["success"]


# ------------------------------------------------------------
# 3. SEPARATE NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

numerical_features = [
    "budget",
    "popularity",
    "runtime",
    "vote_average"
]

categorical_features = [
    "genre"
]


print("\n" + "=" * 60)
print("FEATURE TYPES")
print("=" * 60)

print("Numerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ------------------------------------------------------------
# 4. CREATE ONE-HOT ENCODER
# ------------------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# ------------------------------------------------------------
# 5. ENCODE GENRE
# ------------------------------------------------------------

genre_encoded = encoder.fit_transform(
    X[categorical_features]
)


# ------------------------------------------------------------
# 6. GET ENCODED COLUMN NAMES
# ------------------------------------------------------------

encoded_genre_columns = encoder.get_feature_names_out(
    categorical_features
)


print("\n" + "=" * 60)
print("ENCODED GENRE COLUMNS")
print("=" * 60)

print(encoded_genre_columns)


# ------------------------------------------------------------
# 7. CONVERT ENCODED DATA INTO DATAFRAME
# ------------------------------------------------------------

genre_encoded_df = pd.DataFrame(
    genre_encoded,
    columns=encoded_genre_columns,
    index=X.index
)


# ------------------------------------------------------------
# 8. COMBINE NUMERICAL + ENCODED FEATURES
# ------------------------------------------------------------

X_encoded = pd.concat(
    [
        X[numerical_features],
        genre_encoded_df
    ],
    axis=1
)


# ------------------------------------------------------------
# 9. CHECK FINAL FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ENCODED FEATURE DATASET")
print("=" * 60)

print("Original feature count :", X.shape[1])
print("Encoded feature count :", X_encoded.shape[1])

print("\nFinal feature columns:")
print(X_encoded.columns.tolist())


# ------------------------------------------------------------
# 10. DISPLAY FIRST 5 ROWS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FIRST 5 ENCODED RECORDS")
print("=" * 60)

print(X_encoded.head())


# ------------------------------------------------------------
# 11. CHECK TARGET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TARGET VARIABLE")
print("=" * 60)

print(y.head())

print("\nTarget shape:", y.shape)


# ------------------------------------------------------------
# 12. CHECK FOR MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

print(
    X_encoded.isnull().sum().sum(),
    "missing values in encoded features"
)


# ------------------------------------------------------------
# 13. FINAL STEP 2 SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Genre has been converted from text to numerical features.")
print("Unknown genre has been retained as a valid category.")
print("X_encoded is ready for Train/Test Split.")
