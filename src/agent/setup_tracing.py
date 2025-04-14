import logging
import os

from dotenv import load_dotenv
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager


def setup_environment() -> None:
    load_dotenv()


def setup_tracing() -> None:
    try:
        handler = LlamaIndexCallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host="https://cloud.langfuse.com",
        )
        Settings.callback_manager = CallbackManager([handler])
    except Exception as e:
        logging.error(f"Ошибка Langfuse: {e}")
