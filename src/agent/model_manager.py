import logging

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from agent.llm_wrapper import LLMWrapper
from constants import LLM_NAME, CONTEXT_PATH, CONTEXT_WINDOW, OLLAMA_TIMEOUT


class ModelManager:
    def __init__(self, default_model: str = LLM_NAME):
        self.model_name = default_model
        self.agent = self._init_agent(self.model_name)

    def _init_agent(self, model_name: str) -> LLMWrapper:
        try:
            Settings.llm = Ollama(
                model_name,
                request_timeout=OLLAMA_TIMEOUT,
                context_window=CONTEXT_WINDOW,
            )
            logging.info(f"Инициализация модели {model_name}...")
            return LLMWrapper(llm=Settings.llm, context_file=CONTEXT_PATH)
        except Exception as e:
            logging.error(f"Ошибка при инициализации модели {model_name}: {e}")
            return None

    def switch_model(self, model_name: str) -> None:
        self.model_name = model_name
        self.agent = self._init_agent(model_name)

    def get_agent(self) -> LLMWrapper:
        return self.agent
