# ============================================================
# MOVIEIQ — PHASE 4
# STEP 3: TRAIN / TEST SPLIT
# ============================================================

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ------------------------------------------------------------

df = pd.read_csv("../movies_cleaned.csv")

print("=" * 60)
print("MOVIEIQ — PHASE 4: STEP 3")
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(f"Dataset Shape: {df.shape}")


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


# ------------------------------------------------------------
# 4. ONE-HOT ENCODE GENRE
# ------------------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

genre_encoded = encoder.fit_transform(
    X[categorical_features]
)

encoded_genre_columns = encoder.get_feature_names_out(
    categorical_features
)

genre_encoded_df = pd.DataFrame(
    genre_encoded,
    columns=encoded_genre_columns,
    index=X.index
)


# ------------------------------------------------------------
# 5. CREATE FINAL MODEL FEATURES
# ------------------------------------------------------------

X_encoded = pd.concat(
    [
        X[numerical_features],
        genre_encoded_df
    ],
    axis=1
)


# ------------------------------------------------------------
# 6. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 7. CHECK DATASET SIZES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRAIN / TEST DATASET SIZES")
print("=" * 60)

print(f"Original dataset : {len(df)}")
print(f"Training data    : {len(X_train)}")
print(f"Testing data     : {len(X_test)}")


# ------------------------------------------------------------
# 8. CHECK FEATURE SHAPES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE SHAPES")
print("=" * 60)

print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")

print(f"y_train shape : {y_train.shape}")
print(f"y_test shape  : {y_test.shape}")


# ------------------------------------------------------------
# 9. CHECK TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print("\nTraining set:")
print(y_train.value_counts())
print(
    (y_train.value_counts(normalize=True) * 100).round(2)
)

print("\nTesting set:")
print(y_test.value_counts())
print(
    (y_test.value_counts(normalize=True) * 100).round(2)
)


# ------------------------------------------------------------
# 10. FINAL CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("80% of the data is reserved for training.")
print("20% of the data is reserved for testing.")
print("Stratification preserved the target distribution.")
print("Data is ready for Random Forest training.")
