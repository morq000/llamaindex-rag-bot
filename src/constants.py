# PATHS
PRICELIST_PATH = "/Users/Alexander/Downloads/Прайслист ЧаоЧай.xlsx"
CHROMA_DB_PATH = "./user_data/test_chroma_db"
CHROMA_COLLECTION_NAME = "tea_chroma_collection"
CONTEXT_PATH = "./user_data/user_contexts.json"
LINK_COLUMN_NAME = "Ссылка"
# LLM
CONTEXT_WINDOW = 3900 # Уменьшить окно контекста чтобы избежать timeout (default = 3900)
OLLAMA_TIMEOUT = 100
LLM_NAME = "qwen2.5:14b"
VECTOR_INDEX_TOP_K = 3 # Количество документов для поиска в векторном индексе  
# OTHER
MAX_MESSAGE_LENGTH = 4096  # Максимальная длина сообщения для Telegram
