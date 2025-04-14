import json
import logging
import os
from typing import Dict

from llama_index.core.agent.workflow import AgentOutput, AgentStream, AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.llms import LLM
# from llama_index.llms.ollama import Ollama

from agent.prompts import SYSTEM_PROMPT
from agent.tools import get_tools


class LLMWrapper:
    def __init__(self, llm: LLM, context_file: str = "user_contexts.json"):
        # self.llm = Ollama(model_name, request_timeout=100)
        self.agent = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=get_tools(llm=llm),
            # llm=self.llm,
            system_prompt=SYSTEM_PROMPT,
        )
        self.context_file = context_file
        self.user_contexts: Dict[str, Context] = self._load_contexts()

    async def query(self, user_input: str, user_id: str) -> str:
        try:
            context = self.user_contexts.setdefault(user_id, Context(self.agent))
            handler = self.agent.run(user_msg=user_input, ctx=context)

            result = ""
            async for event in handler.stream_events():
                try:
                    if isinstance(event, AgentStream):
                        result += event.delta
                    elif isinstance(event, AgentOutput) and event.response.content:
                        result = event.response.content
                except Exception as e:
                    logging.error(f"Ошибка при обработке события: {e}")
                    continue

            self._save_contexts()
            return result

        except Exception as e:
            logging.exception("Ошибка при выполнении запроса")
            return f"Ошибка при выполнении запроса: {str(e)}"

    async def close(self) -> None:
        self._save_contexts()

    def _load_contexts(self) -> Dict[str, Context]:
        if not os.path.exists(self.context_file):
            return {}

        try:
            with open(self.context_file, "r", encoding="utf-8") as file:
                data = json.load(file)
                return {
                    user_id: Context.from_dict(self.agent, ctx) for user_id, ctx in data.items()
                }
        except Exception as e:
            logging.warning(f"Не удалось загрузить контексты: {e}")
            return {}

    def _save_contexts(self) -> None:
        try:
            with open(self.context_file, "w", encoding="utf-8") as file:
                data = {user_id: ctx.to_dict() for user_id, ctx in self.user_contexts.items()}
                json.dump(data, file, ensure_ascii=False, indent=4)
        except Exception as e:
            logging.error(f"Ошибка при сохранении контекста: {e}")
