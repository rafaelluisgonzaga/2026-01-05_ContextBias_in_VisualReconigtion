from dotenv import load_dotenv
import os

load_dotenv()

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
UNSPLASH_SECRET_KEY = os.getenv("UNSPLASH_SECRET_KEY")

if not UNSPLASH_ACCESS_KEY:
    raise ValueError("UNSPLASH_ACCESS_KEY não encontrada no .env")

if not UNSPLASH_SECRET_KEY:
    raise ValueError("UNSPLASH_SECRET_KEY não encontrada no .env")