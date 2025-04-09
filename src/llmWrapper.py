import json
import os
from llama_index.core.agent.workflow import AgentWorkflow, AgentStream, AgentOutput
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context

# from llama_index.core import PromptTemplate
from prompts import SYSTEM_PROMPT
from tools import get_tools


class OllamaWrapper:
    def __init__(self, model_name: str, context_file: str = "user_contexts.json"):
        # Инициализация LLM с указанной моделью
        self.llm = Ollama(model_name, request_timeout=100)
        # Создаём агент с пустым списком инструментов, передаём llm и системный промпт
        self.agent = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=get_tools(),
            llm=self.llm,
            system_prompt="\n".join(SYSTEM_PROMPT),
        )
        self.context_file = context_file
        self.user_contexts = self.load_contexts()

    def load_contexts(self):
        if os.path.exists(self.context_file):
            with open(self.context_file, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    return {
                        user_id: Context.from_dict(self.agent, ctx_data)
                        for user_id, ctx_data in data.items()
                    }
                except json.JSONDecodeError:
                    print(
                        "Ошибка при декодировании JSON. Файл контекста может быть повреждён."
                    )
        return {}

    def save_contexts(self):
        with open(self.context_file, "w", encoding="utf-8") as file:
            data = {
                user_id: ctx.to_dict() for user_id, ctx in self.user_contexts.items()
            }
            json.dump(data, file, ensure_ascii=False, indent=4)

    async def query(self, user_input: str, user_id: str) -> str:
        try:
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = Context(self.agent)
            user_ctx = self.user_contexts[user_id]

            handler = self.agent.run(user_msg=user_input, ctx=user_ctx)
            result = ""
            async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                    result += event.delta
                elif isinstance(event, AgentOutput) and event.response.content:
                    result = event.response.content

            self.save_contexts()
            return result
        except Exception as e:
            return f"Ошибка при выполнении запроса: {str(e)}"

    async def close(self):
        self.save_contexts()
