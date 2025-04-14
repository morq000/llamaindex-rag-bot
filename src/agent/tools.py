import logging

import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.tools import QueryEngineTool
from llama_index.experimental import PandasQueryEngine

from constants import LINK_COLUMN_NAME, VECTOR_INDEX_TOP_K
from utils.pricelist_retriever import retrieve_xlsx
from utils.vector_index_builder import build_index

# def get_visit_webpage_tool():
#     return FunctionTool.from_defaults(
#         fn=fetch_page_content,
#         name="fetch_page_content",
#         description="Используй этот инструмент, чтобы получить HTML-код страницы по заданному URL."
#     )


# Internal methods
def _get_pd_query_engine_tool(df: pd.DataFrame, llm: LLM, verbose: bool = False) -> QueryEngineTool:
    pd_query_engine = PandasQueryEngine(df, llm=llm, verbose=verbose)
    # print('Query engine prompts: ', pd_query_engine.get_prompts())
    return QueryEngineTool.from_defaults(
        query_engine=pd_query_engine,
        name="pandas_query_engine",
        description="Pandas query engine tool. Содержит прайс-лист чая в виде Dataframe. Используй для получения информации о наличии чая и его стоимости.",
    )


def _get_query_engine_tool_from_index(index: VectorStoreIndex) -> QueryEngineTool:
    return QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(similarity_top_k=VECTOR_INDEX_TOP_K),
        name="vector_index_query_engine",
        description="Векторая база с описаниями всех видов чая. Используй для получения описания чая в случае если пользователь спрашивает о конкретном сорте чая.",
    )


# External methods
# Function to get all tools
def get_tools(llm: LLM) -> list[QueryEngineTool]:
    tools = []
    # Get pd query engine tool
    df = retrieve_xlsx()
    pd_query_engine_tool = _get_pd_query_engine_tool(df, llm, True)
    tools.append(pd_query_engine_tool)
    logging.info("Tool added: " + pd_query_engine_tool.metadata.description)

    # Get vector index tool
    index = build_index(df, LINK_COLUMN_NAME)
    if index:
        index_query_engine_tool = _get_query_engine_tool_from_index(index)
        logging.info("Tool added: " + index_query_engine_tool.metadata.description)
        tools.append(index_query_engine_tool)
    return tools
