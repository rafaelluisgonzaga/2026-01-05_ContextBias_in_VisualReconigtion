import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "data/processed/dataset.csv"
FEEDBACK_PATH = "data/processed/feedback.csv"

st.set_page_config(page_title="Human-in-the-Loop City Guess", layout="centered")
st.title("🧠 Human-in-the-Loop — Escolha a cidade correta")

# -----------------------------
# LOAD MODEL + DATA
# -----------------------------
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv(DATASET_PATH)

    y = df["label"]
    X = df[["brightness", "region"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["brightness"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["region"]),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X, y)
    return model, df


model, df = load_model_and_data()

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "remaining_indices" not in st.session_state:
    st.session_state.remaining_indices = list(df.index)

if "current_idx" not in st.session_state and st.session_state.remaining_indices:
    st.session_state.current_idx = st.session_state.remaining_indices.pop(0)

# -----------------------------
# END CONDITION
# -----------------------------
if "current_idx" not in st.session_state:
    st.success("🎉 Todas as imagens foram analisadas!")
    st.markdown(
        f"""
        ### ✅ Processo finalizado
        
        - Todas as imagens receberam feedback humano
        - Arquivo salvo em: `{FEEDBACK_PATH}`
        - Pronto para análise ou re-treino
        """
    )
    st.stop()

# -----------------------------
# CURRENT SAMPLE
# -----------------------------
sample = df.loc[st.session_state.current_idx]
img_path = sample["image_path"]

if not os.path.exists(img_path):
    st.error(f"Imagem não encontrada:\n{img_path}")
    st.stop()

img = Image.open(img_path)
filename = os.path.basename(img_path)

# -----------------------------
# TOP-3 PREDICTION
# -----------------------------
X_sample = sample[["brightness", "region"]].to_frame().T
proba = model.predict_proba(X_sample)[0]
classes = model.classes_

top3_idx = np.argsort(proba)[-3:][::-1]
top3_labels = classes[top3_idx]


def label_to_city(label):
    return df[df["label"] == label]["image_path"].iloc[0].split("/")[-2]


top3_cities = [label_to_city(l) for l in top3_labels]
true_city = label_to_city(sample["label"])

# -----------------------------
# UI
# -----------------------------
st.image(
    img,
    caption=f"Imagem analisada — {filename}",
    use_container_width=True
)

choice = st.radio(
    "Qual cidade é essa?",
    top3_cities,
    key="choice"
)

# -----------------------------
# CONFIRM BUTTON
# -----------------------------
if st.button("Confirmar"):
    correct = choice == true_city

    # salvar feedback
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    with open(FEEDBACK_PATH, "a") as f:
        f.write(f"{img_path},{choice},{true_city},{correct}\n")

    if correct:
        st.success("✅ Correto!")
    else:
        st.error(f"❌ Errado. Correta: {true_city}")

    # avançar para próxima imagem
    if st.session_state.remaining_indices:
        st.session_state.current_idx = st.session_state.remaining_indices.pop(0)
    else:
        del st.session_state.current_idx

    st.rerun()