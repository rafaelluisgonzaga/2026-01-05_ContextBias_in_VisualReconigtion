import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "data/processed/dataset.csv"


def run():
    # 1. Carregar dados
    df = pd.read_csv(DATASET_PATH)

    # 2. Label real (cidade)
    y = df["label"]

    # 3. Features SEM vazamento de identidade
    X = df[
        [
            "brightness",  # aparência da imagem
            "region",      # contexto amplo (não único)
        ]
    ]

    # 4. Tipos de variáveis
    numeric_features = ["brightness"]
    categorical_features = ["region"]

    # 5. Pré-processamento
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # 6. Pipeline
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # 7. Split treino / teste (natural)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # 8. Treinar
    model.fit(X_train, y_train)

    # 9. Avaliar
    y_pred = model.predict(X_test)

    print("\n📈 RESULTADO DO TREINO (SEM VAZAMENTO DE IDENTIDADE)")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}\n")
    print("📊 Classification report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    run()