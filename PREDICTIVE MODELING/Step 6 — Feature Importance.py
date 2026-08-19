# ============================================================
# MOVIEIQ — PHASE 4
# STEP 6: FEATURE IMPORTANCE
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ------------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ------------------------------------------------------------

df = pd.read_csv("../movies_cleaned.csv")

print("=" * 60)
print("MOVIEIQ — PHASE 4: STEP 6")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)


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
# 3. DEFINE NUMERICAL AND CATEGORICAL FEATURES
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


# ------------------------------------------------------------
# 7. TRAIN RANDOM FOREST
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 8. EXTRACT FEATURE IMPORTANCE
# ------------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})


# ------------------------------------------------------------
# 9. SORT FEATURES
# ------------------------------------------------------------

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


# ------------------------------------------------------------
# 10. DISPLAY FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance.to_string(index=False)
)


# ------------------------------------------------------------
# 11. TOP 10 FEATURES
# ------------------------------------------------------------

top_features = feature_importance.head(10)

print("\n" + "=" * 60)
print("TOP 10 FEATURES")
print("=" * 60)

print(
    top_features.to_string(index=False)
)


# ------------------------------------------------------------
# 12. VISUALIZE FEATURE IMPORTANCE
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "../assets/09_random_forest_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 13. COMPLETION MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PHASE 4 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Random Forest training completed.")
print("Model evaluation completed.")
print("Feature importance analysis completed.")
