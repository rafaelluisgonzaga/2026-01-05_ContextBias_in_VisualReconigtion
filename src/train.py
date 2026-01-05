import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "data/processed/dataset.csv"
FEEDBACK_PATH = "data/processed/feedback.csv"


def run():
    # 1. Carregar dataset principal
    df = pd.read_csv(DATASET_PATH)

    # 2. Carregar feedback humano
    feedback = pd.read_csv(
        FEEDBACK_PATH,
        names=["image_path", "chosen_city", "true_city", "correct"]
    )

    # 3. Criar pesos (default = 1.0)
    df["sample_weight"] = 1.0

    # 4. Aumentar peso das imagens erradas pelo modelo
    wrong_images = feedback[feedback["correct"] == False]["image_path"]

    df.loc[df["image_path"].isin(wrong_images), "sample_weight"] = 2.0

    # 5. Labels reais
    y = df["label"]

    # 6. Features SEM vazamento
    X = df[["brightness", "region"]]

    # 7. Tipos de variáveis
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

    # 8. Split treino / teste
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        df["sample_weight"],
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # 9. Treinar com pesos
    model.fit(
        X_train,
        y_train,
        classifier__sample_weight=w_train
    )

    # 10. Avaliar
    y_pred = model.predict(X_test)

    print("\n📈 RESULTADO DO TREINO (COM FEEDBACK HUMANO)")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}\n")
    print("📊 Classification report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    run()