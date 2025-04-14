# LlamaIndex - based RAG chatbot

## Description
This is a pet project chatbot that uses LlamaIndex and Aiogram libs to create a Telegram bot capable of answering questions about tea store goods and tea descriptions. The pricelist is a real tea store pricelist with tea descriptions and goods quantity data. 
- The bot tracks context for each user.
- LangFuse is used for bot observability and monitoring.
- Ollama models are used for LLMs.

The agent workflow has two tools:
1. Query engine tool based on Pandas DataFrame, which is generated from XLS pricelist file. This one is used to answer questions about tea store goods and their quantity. 
2. Query engine tool based Vector store (ChromaDB) using embeddings which is generated from URLs with tea descriptions. The Vector store is used to answer questions about tea descriptions.

## Design
The architecture of the bot is based on LlamaIndex and Aiogram libraries. LlamaIndex uses local Ollama models. Bot has command to switch underlying model to use. Aiogram is used to create a Telegram bot that interacts with users and handles incoming messages. The bot uses LlamaIndex to process user queries and generate responses based on tool retrieved data.

## Further improvements
- Add more tools to the agent workflow to improve the bot's capabilities:
    - use Google calendar API to provide data about upcoming events
    - generate an order from pricelist based on user query
- Add voice recognition and answering
- Try to use OpenAI API to generate more complex queries