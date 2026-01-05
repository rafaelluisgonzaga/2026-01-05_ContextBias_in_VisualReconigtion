import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

DATASET_PATH = "data/processed/dataset.csv"


def top_k_accuracy(y_true, y_proba, k):
    top_k = np.argsort(y_proba, axis=1)[:, -k:]
    hits = [
        y_true.iloc[i] in top_k[i]
        for i in range(len(y_true))
    ]
    return np.mean(hits)


def run():
    df = pd.read_csv(DATASET_PATH)

    y = df["label"]
    X = df[["brightness", "region"]]

    numeric_features = ["brightness"]
    categorical_features = ["region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    # Predições
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n🎯 AVALIAÇÃO TOP-K")
    print(f"Top-1 Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Top-3 Accuracy: {top_k_accuracy(y_test, y_proba, k=3):.2f}")
    print(f"Top-5 Accuracy: {top_k_accuracy(y_test, y_proba, k=5):.2f}")

    return y_test, y_pred, X_test


if __name__ == "__main__":
    run()