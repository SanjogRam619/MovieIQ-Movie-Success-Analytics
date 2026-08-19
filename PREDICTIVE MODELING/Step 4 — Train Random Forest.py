# ============================================================
# MOVIEIQ — PHASE 4
# STEP 4: RANDOM FOREST TRAINING
# ============================================================

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ------------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ------------------------------------------------------------

df = pd.read_csv("../movies_cleaned.csv")

print("=" * 60)
print("MOVIEIQ — PHASE 4: STEP 4")
print("RANDOM FOREST TRAINING")
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
# 5. CREATE FINAL FEATURE DATASET
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


print("\n" + "=" * 60)
print("DATA READY FOR TRAINING")
print("=" * 60)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")


# ------------------------------------------------------------
# 7. CREATE RANDOM FOREST MODEL
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ------------------------------------------------------------
# 8. TRAIN THE MODEL
# ------------------------------------------------------------

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully!")


# ------------------------------------------------------------
# 9. BASIC MODEL INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RANDOM FOREST MODEL")
print("=" * 60)

print(model)


# ------------------------------------------------------------
# 10. STEP 4 COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Random Forest has been trained.")
print("The model is now ready to make predictions.")
