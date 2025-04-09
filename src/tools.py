import pandas as pd
import logging
from llama_index.experimental import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from constants import LLM_NAME, LINK_COLUMN_NAME
from pricelistRetriever import retrieve_xlsx
from utils import build_index, get_docs_from_dataframe
    
# def get_visit_webpage_tool():
#     return FunctionTool.from_defaults(
#         fn=fetch_page_content,
#         name="fetch_page_content",
#         description="Используй этот инструмент, чтобы получить HTML-код страницы по заданному URL."
#     )

# Tool functions
def get_pd_query_engine_tool(df: pd.DataFrame, llm: LLM, verbose: bool = False):
    pd_query_engine = PandasQueryEngine(df, llm=llm, verbose=verbose)
    # print('Query engine prompts: ', pd_query_engine.get_prompts())
    return QueryEngineTool.from_defaults(
        query_engine=pd_query_engine,
        name="tea_pricelist",
        description="Прайс-лист всех видов чая в ниличии. Используй для получения информации о наличии чая и его стоимости.",
    )

def get_query_engine_tool_from_index(index: VectorStoreIndex):
    return QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(),
        name="tea_index",
        description="Векторая база с описаниями всех видов чая. Используй для получения описания чая в случае если пользователь спрашивает о конкретном сорте чая.",
    )

# Function to get all tools
def get_tools():
    tools = []
    df = retrieve_xlsx()
    pd_query_engine_tool = get_pd_query_engine_tool(df, LLM_NAME, verbose=True)
    tools.append(pd_query_engine_tool)
    logging.info(pd_query_engine_tool.metadata.description)

    index = build_index(df, LINK_COLUMN_NAME)
    if index:
        index_query_engine_tool = get_query_engine_tool_from_index(index)
        logging.info(index_query_engine_tool.metadata.description)
        tools.append(index_query_engine_tool)
    return tools
