import logging

from aiogram import Router, types
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from agent.model_manager import ModelManager
from constants import MAX_MESSAGE_LENGTH

router = Router()

model_mapping = {
    "model_deepseek": "deepseek-r1:8b",
    "model_llama": "llama3.2:latest",
    "model_mistral": "mistral:latest",
    "model_qwen": "qwen2.5:14b",
}


def get_model_keyboard() -> types.InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for key, value in model_mapping.items():
        builder.add(types.InlineKeyboardButton(text=value, callback_data=key))
    return builder.as_markup()


def split_message(text: str) -> list[str]:
    return [text[i : i + MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)]


def register_handlers(router: Router, model_manager: ModelManager) -> None:
    @router.message(Command("model"))
    async def cmd_model(message: Message) -> None:
        keyboard = get_model_keyboard()
        await message.answer("Выберите модель для генерации текста:", reply_markup=keyboard)

    @router.callback_query(lambda c: c.data.startswith("model_"))
    async def handle_model_selection(callback_query: CallbackQuery) -> None:
        model_key = callback_query.data
        if not model_key:
            await callback_query.answer("Ошибка: модель не выбрана.")
            return
        model_name = model_mapping.get(model_key)
        try:
            model_manager.switch_model(model_name)
            await callback_query.answer(f"Вы выбрали модель: {model_name}")
        except Exception as e:
            logging.error(f"Ошибка при выборе модели {model_name}: {e}")
            await callback_query.answer("Ошибка при выборе модели.")

    @router.message()
    async def handle_message(message: Message) -> None:
        agent = model_manager.get_agent()
        if not agent:
            await message.answer("Модель не выбрана. Используйте /model.")
            return
        try:
            if not message.from_user:
                logging.error("Ошибка: не удалось получить информацию о пользователе.")
                return
            response = await agent.query(message.text, message.from_user.id)
            for msg in split_message(response):
                await message.answer(msg)
        except Exception as e:
            logging.error(f"Ошибка генерации: {e}")
            await message.answer("Ошибка при генерации ответа.")
