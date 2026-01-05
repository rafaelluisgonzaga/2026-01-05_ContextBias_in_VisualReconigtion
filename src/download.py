import os
import requests
from config import UNSPLASH_ACCESS_KEY

BASE_URL = "https://api.unsplash.com/search/photos"

CITIES = {
    "paris": ["Paris city day", "Paris city night"],
    "london": ["London city day", "London city night"],
    "rome": ["Rome city day", "Rome city night"],
    "berlin": ["Berlin city day", "Berlin city night"],
    "madrid": ["Madrid city sunny day", "Madrid city night lights"],
    "amsterdam": ["Amsterdam city sunny day", "Amsterdam city night lights"],
    "vienna": ["Vienna city sunny day", "Vienna city night lights"],
    "prague": ["Prague city sunny day", "Prague city night lights"],
}

IMAGES_PER_QUERY = 10
OUTPUT_DIR = "data/raw"


def fetch_images(query, per_page=10):
    headers = {
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
    }

    params = {
        "query": query,
        "per_page": per_page,
        "orientation": "landscape",
    }

    response = requests.get(BASE_URL, headers=headers, params=params)
    response.raise_for_status()
    return response.json()["results"]


def download_image(url, path):
    if os.path.exists(path):
        return False

    img = requests.get(url, stream=True)
    img.raise_for_status()

    with open(path, "wb") as f:
        for chunk in img.iter_content(1024):
            f.write(chunk)

    return True


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for city, queries in CITIES.items():
        city_dir = os.path.join(OUTPUT_DIR, city)

        # 🔒 SE A PASTA EXISTE E NÃO ESTÁ VAZIA → PULA A CIDADE
        if os.path.exists(city_dir) and os.listdir(city_dir):
            print(f"\n⏭ Pulando {city.upper()} (já existe)")
            continue

        os.makedirs(city_dir, exist_ok=True)

        print(f"\n📥 Baixando imagens de {city.upper()}")

        counter = 1  # 🔢 contador POR CIDADE

        for query in queries:
            results = fetch_images(query, IMAGES_PER_QUERY)

            for photo in results:
                filename = f"{city}_{counter}.jpg"
                filepath = os.path.join(city_dir, filename)

                baixou = download_image(photo["urls"]["regular"], filepath)

                if baixou:
                    print(f"  ✔ {filename}")
                    counter += 1
                else:
                    print(f"  ↪ já existe: {filename}")


if __name__ == "__main__":
    run()