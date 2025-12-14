import os
from dotenv import load_dotenv

load_dotenv(os.path.expanduser('~/.llm_env'))
API_KEY : str= os.getenv("MY_OPENAI_API_KEY") or ""
BASE_URL = os.getenv("MY_OPENAI_API_BASE")


