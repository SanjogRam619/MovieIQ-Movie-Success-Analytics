# ============================================================
# MOVIEIQ — PHASE 4
# STEP 5: PREDICTIONS & MODEL EVALUATION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ------------------------------------------------------------

df = pd.read_csv("../movies_cleaned.csv")

print("=" * 60)
print("MOVIEIQ — PHASE 4: STEP 5")
print("PREDICTIONS & MODEL EVALUATION")
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
# 3. SEPARATE FEATURES
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
# 7. CREATE RANDOM FOREST
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ------------------------------------------------------------
# 8. TRAIN MODEL
# ------------------------------------------------------------

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully!")


# ------------------------------------------------------------
# 9. MAKE PREDICTIONS
# ------------------------------------------------------------

print("\nMaking predictions on test data...")

y_pred = model.predict(X_test)

print("Predictions completed successfully!")


# ------------------------------------------------------------
# 10. CALCULATE ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# 11. CALCULATE PRECISION
# ------------------------------------------------------------

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)


# ------------------------------------------------------------
# 12. CALCULATE RECALL
# ------------------------------------------------------------

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


# ------------------------------------------------------------
# 13. DISPLAY PERFORMANCE METRICS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")


# ------------------------------------------------------------
# 14. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Successful",
            "Successful"
        ],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 15. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ------------------------------------------------------------
# 16. VISUALIZE CONFUSION MATRIX
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Not Successful",
        "Successful"
    ],
    yticklabels=[
        "Not Successful",
        "Successful"
    ]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "../assets/08_random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 17. BASELINE ACCURACY
# ------------------------------------------------------------

baseline_accuracy = (
    y_test.value_counts(normalize=True).max()
)

print("\n" + "=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)

print(
    f"Baseline Accuracy : "
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    f"Random Forest     : "
    f"{accuracy * 100:.2f}%"
)


if accuracy > baseline_accuracy:

    print("\nRandom Forest performs better than the baseline.")

elif accuracy == baseline_accuracy:

    print("\nRandom Forest matches the baseline.")

else:

    print("\nRandom Forest performs below the baseline.")


# ------------------------------------------------------------
# 18. STEP 5 COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("The Random Forest has been evaluated.")
print("Predictions, performance metrics and confusion matrix")
print("have been generated.")
