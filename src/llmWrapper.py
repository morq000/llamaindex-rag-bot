import json
import os
from llama_index.core.agent.workflow import AgentWorkflow, AgentStream, AgentOutput, ToolCall, ToolCallResult
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from composio_openai import ComposioToolSet, App

composio_toolset = ComposioToolSet(api_key="e9wewgtvu9pezcf0thm4rm", entity_id="default")
tools = composio_toolset.get_tools(apps=[App.GOOGLESHEETS])


class OllamaWrapper:
    def __init__(self, model_name: str, context_file: str = 'user_contexts.json'):
        # Инициализация LLM с указанной моделью
        self.llm = Ollama(model_name, request_timeout=100)
        # Создаём агент с пустым списком инструментов, передаём llm и системный промпт
        self.agent = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=tools,
            llm=self.llm,
            system_prompt=(
                "Ты — консультант чайного магазина (женщина) с лёгким китайским акцентом."
                "Ты можешь рассказывать только о китайском чае, его сортах, истории, традициях, вкусовых особенностях и рекомендовать, какой сорт выбрать."
                "Ты можешь отвечать на вопросы, связанные с ассортиментом и общими сведениями о чае."
                "Если чай упоминается в контексте поьзы для здоровья, можно ответить ощими фразами, но не углубляться в тему, и не давать медицинских советов."
                "Если вопрос касается политики, религии, финансов (вне цен и оплаты в магазине), медицины, сексуальных тем или других спорных вопросов — даже если там упоминается чай, — ты должна отвечать: «Чайные листья не дают ответа на этот вопрос.»"
                "Если вопрос не связан с чаем или не является приветствием, ты также отвечаешь: «Чайные листья не дают ответа на этот вопрос.»"
                "Если пользователь просто здоровается, поздоровайся в ответ и можешь предложить помощь в выборе чая."
                "Не используй гендерные местоимения и прилагательные, если не уверена в том, какой у пользователя пол."
                "Всегда сохраняй дружелюбный тон."
                "Если вопрос связан с ценами в магазине, способами доставки или способами заваривания, отвечай развернуто."
                "Во всех других случаях — «Чайные листья не дают ответа на этот вопрос.»"
                "Ты можешь использовать инструменты для просмотра Google Sheets, но не можешь их редактировать."
                "Если тебя спрашивают, какой чай есть в магазине, используй файл 'Прайслист ЧаоЧай'"
            )
        )
        self.context_file = context_file
        self.user_contexts = self.load_contexts()

    
    def load_contexts(self):
        if os.path.exists(self.context_file):
            with open(self.context_file, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    return {user_id: Context.from_dict(self.agent, ctx_data) for user_id, ctx_data in data.items()}
                except json.JSONDecodeError:
                    print("Ошибка при декодировании JSON. Файл контекста может быть повреждён.")
        return {}

    def save_contexts(self):
        with open(self.context_file, 'w', encoding='utf-8') as file:
            data = {user_id: ctx.to_dict() for user_id, ctx in self.user_contexts.items()}
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
                elif isinstance(event, ToolCall):
                    result = composio_toolset.handle_tool_calls(event.response)
                    print(result)

            self.save_contexts()
            return result
        except Exception as e:
            return f"Ошибка при выполнении запроса: {str(e)}"

    async def close(self):
        self.save_contexts()
