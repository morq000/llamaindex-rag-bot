import chromadb
import pandas as pd
import logging
import requests
import html2text
from llama_index.core import Document, VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from constants import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME

# Функция для создания или получения коллекции
def get_or_create_collection(db: chromadb.PersistentClient, collection_name: str):
    try:
        # Пытаемся получить существующую коллекцию
        return db.get_collection(collection_name)
    except ValueError:
        # Если коллекция не существует, создаем новую
        return db.create_collection(collection_name)

# Функция для построения индекса
def build_index(df: pd.DataFrame, linkColumnName: str) -> VectorStoreIndex:
    # Создаем клиент для работы с ChromaDB
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Получаем или создаем коллекцию
    chroma_collection = get_or_create_collection(db, CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Используем модель HuggingFace для создания эмбеддингов
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    # Проверяем, была ли коллекция только что создана
    if chroma_collection.count() == 0:
        logging.info("Создана новая коллекция: %s", CHROMA_COLLECTION_NAME)
        documents = get_docs_from_dataframe(df, linkColumnName)
        if not documents:
            logging.error("Нет документов для построения индекса")
            return None
        # Создаем пайплайн для обработки документов
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(),
                embed_model,
            ],
            vector_store=vector_store
        )
        # Строим индекс
        try:
            pipeline.run(documents=documents)
            index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
            logging.info("Индекс успешно создан")
        except Exception as e:
            logging.error("Ошибка при создании индекса: %s", e)
            return None
    else:
        logging.info("Используется существующая коллекция: %s", CHROMA_COLLECTION_NAME)
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    return index

def get_docs_from_dataframe(df: pd.DataFrame, linkCoulumnName: str):
    links = []
    for index, row in df.iterrows():
        url = row[linkCoulumnName]
        if url:
            links.append(url)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        documents = load_data(links, headers)
        logging.info(f"Загружено {len(documents)} документов")
    except Exception as e:
        logging.error("Ошибка при загрузке документов: %s", e)
    return documents

def load_data(urls: list[str], headers: dict) -> list[Document]:
    documents = []
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            text = html2text.html2text(response.text)
            documents.append(Document(text=text, id_=url))
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 404:
                logging.error("Страница не найдена (404) для URL: %s", url)
            else:
                logging.error("HTTP ошибка для URL %s: %s", url, http_err)
        except Exception as e:
            logging.error("Ошибка при загрузке документа с URL %s: %s", url, e)
    return documents

# def fetch_page_content(url: str) -> str:
#     """Получает содержимое страницы по URL.
#         Args:
#             url: URL страницы.
#     """
#     try:
#         headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
#         }
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
#         content = response.text 
#         return content
#     except requests.RequestException as e:
#         return f"Ошибка при получении страницы: {str(e)}"