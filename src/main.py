import asyncio
import os

from aiogram import Bot, Dispatcher

from agent.model_manager import ModelManager
from agent.setup_tracing import setup_environment, setup_tracing
from bot.handlers import register_handlers, router
from utils.logging import setup_logging


async def main() -> None:
    # Setup environment
    setup_environment()
    setup_logging()
    setup_tracing()

    # Setup bot and dispatcher
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not isinstance(token, str):
        print("Error: TELEGRAM_BOT_TOKEN is not set or invalid.")
        exit(1)
    bot = Bot(token=token)
    dp = Dispatcher()
    model_manager = ModelManager()
    if not model_manager.agent:
        print("Error: Model agent is not initialized.")
        exit(1)

    register_handlers(router, model_manager)
    dp.include_router(router)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
