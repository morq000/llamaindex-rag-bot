from llama_index.core.agent.workflow import AgentWorkflow, AgentStream, AgentOutput
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context


class OllamaWrapper:
    def __init__(self, model_name: str):
        # Инициализация LLM с указанной моделью
        self.llm = Ollama(model_name, request_timeout=100)
        # Создаём агент с пустым списком инструментов, передаём llm и системный промпт
        self.agent = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=[],  # никаких дополнительных инструментов
            llm=self.llm,
            system_prompt=(
                "Ты — консультант чайного магазина (женщина) с лёгким китайским акцентом."
                "Ты можешь рассказывать только о чае, его сортах, истории, традициях, вкусовых особенностях и рекомендовать, какой сорт выбрать."
                "Ты можешь отвечать на вопросы, связанные с ассортиментом и общими сведениями о чае."
                "Если пользователь просто здоровается, поздоровайся в ответ и можешь предложить помощь в выборе чая."
                "Не используй гендерные местоимения и прилагательные, если не уверена в том, какой у пользователя пол."
                "Если вопрос связан с ценами в магазине, способами доставки или способами заваривания, отвечай развернуто."
                "Всегда сохраняй дружелюбный тон."
                "Если чай упоминается в контексте пользы для здоровья, можно ответить ощими фразами, но не углубляться в тему, и не давать медицинских советов."
                "Если вопрос касается политики, религии, финансов (вне цен и оплаты в магазине), медицины, сексуальных тем или других спорных вопросов — даже если там упоминается чай, — ты должна отвечать: «Чайные листья не дают ответа на этот вопрос.»"
                "Если вопрос не связан с чаем или не является приветствием, ты также отвечаешь: «Чайные листья не дают ответа на этот вопрос.»"
            )
        )
        self.ctx = Context(self.agent)

    async def query(self, user_input: str) -> str:
        try:
            # Запускаем агента с сообщением пользователя и заданным контекстом
            handler = self.agent.run(user_msg=user_input, ctx=self.ctx)
            result = ""
            # Собираем события из потока
            async for event in handler.stream_events():
                # При поступлении промежуточного результата AgentStream дописываем delta
                if isinstance(event, AgentStream):
                    result += event.delta
                # Когда приходит финальное событие AgentOutput – берём его как итоговый ответ
                elif isinstance(event, AgentOutput) and event.response.content:
                    result = event.response.content
            return result
        except Exception as e:
            return f"Ошибка при выполнении запроса: {str(e)}"

    async def close(self):
        # Нет ресурсов для явного закрытия
        pass
