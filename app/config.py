import os
from dotenv import load_dotenv

load_dotenv() #.env파일의 환경 변수를 읽어 현재 프로세서의 환경 변수를 로드함.

class Config:
    UPLOAD_DIR = os.getenv('UPLOAD_DIR', './data/uploads')
    CHROMA_DIR = os.getenv('CHROMA_DIR', './data/chroma')
    SAVE_PATH = os.getenv('SAVE_PATH', './data/logs')
    SAVE_OPTION = os.getenv('SAVE_OPTION', False)
    EMBED_MODEL = os.getenv("EMBED_MODEL", 'upskyy/bge-m3-korean')
    VL_MODEL = os.getenv("VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct")  # or Qwen/Qwen3-VL-8B-Instruct

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")