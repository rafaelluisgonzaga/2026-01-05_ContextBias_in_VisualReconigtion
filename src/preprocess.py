import os
import csv
from PIL import Image
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
METADATA_PATH = "data/metadata/cities_metadata.csv"
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "dataset.csv")


def calcular_brilho(path):
    img = Image.open(path).convert("L")  # grayscale
    arr = np.array(img)
    return arr.mean()


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 🔹 carregar metadata das cidades
    metadata = pd.read_csv(METADATA_PATH)
    metadata.set_index("city", inplace=True)

    rows = []
    label_map = {}
    label_id = 0

    for city in os.listdir(RAW_DIR):
        city_path = os.path.join(RAW_DIR, city)
        if not os.path.isdir(city_path):
            continue

        if city not in metadata.index:
            print(f"⚠️ Cidade '{city}' sem metadata — ignorada")
            continue

        if city not in label_map:
            label_map[city] = label_id
            label_id += 1

        label = label_map[city]

        city_meta = metadata.loc[city]

        for file in os.listdir(city_path):
            if not file.lower().endswith(".jpg"):
                continue

            img_path = os.path.join(city_path, file)
            brilho = calcular_brilho(img_path)

            rows.append([
                img_path,
                brilho,
                city_meta["lat"],
                city_meta["lon"],
                city_meta["population"],
                city_meta["region"],
                label
            ])

    # 🔹 salvar CSV final
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "brightness",
            "lat",
            "lon",
            "population",
            "region",
            "label"
        ])
        writer.writerows(rows)

    print(f"✅ Dataset gerado em: {OUTPUT_CSV}")
    print("📌 Labels:", label_map)


if __name__ == "__main__":
    run()