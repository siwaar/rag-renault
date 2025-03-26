"""
  Renault Agent with different tools 
"""
from langchain_core.tools import tool
from datetime import date
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
import yfinance as yf
from rag_app import get_response_with_sources
from retriever import get_retriever
from config.settings import model


@tool
def company_information(ticker: str) -> dict:
    """Use this tool to retrieve company information like address, industry, sector, company officers, business summary, website,
    marketCap, current price, ebitda, total debt, total revenue, debt-to-equity, etc."""

    ticker_obj = yf.Ticker(ticker)
    ticker_info = ticker_obj.get_info()

    return ticker_info


@tool
def last_dividend_and_earnings_date(ticker: str) -> dict:
    """
    Use this tool to retrieve company's last dividend date and earnings release dates.
    It does not provide information about historical dividend yields.
    """
    ticker_obj = yf.Ticker(ticker)

    return ticker_obj.get_calendar()


@tool
def summary_of_mutual_fund_holders(ticker: str) -> dict:
    """
    Use this tool to retrieve company's top mutual fund holders.
    It also returns their percentage of share, stock count and value of holdings.
    """
    ticker_obj = yf.Ticker(ticker)
    mf_holders = ticker_obj.get_mutualfund_holders()

    return mf_holders.to_dict(orient="records")


@tool
def summary_of_institutional_holders(ticker: str) -> dict:
    """
    Use this tool to retrieve company's top institutional holders.
    It also returns their percentage of share, stock count and value of holdings.
    """
    ticker_obj = yf.Ticker(ticker)
    inst_holders = ticker_obj.get_institutional_holders()

    return inst_holders.to_dict(orient="records")


@tool
def stock_grade_updrages_downgrades(ticker: str) -> dict:
    """
    Use this to retrieve grade ratings upgrades and downgrades details of particular stock.
    It'll provide name of firms along with 'To Grade' and 'From Grade' details. Grade date is also provided.
    """
    ticker_obj = yf.Ticker(ticker)

    curr_year = date.today().year

    upgrades_downgrades = ticker_obj.get_upgrades_downgrades()
    upgrades_downgrades = upgrades_downgrades.loc[
        upgrades_downgrades.index > f"{curr_year}-01-01"
    ]
    upgrades_downgrades = upgrades_downgrades[
        upgrades_downgrades["Action"].isin(["up", "down"])
    ]

    return upgrades_downgrades.to_dict(orient="records")


@tool
def stock_splits_history(ticker: str) -> dict:
    """
    Use this tool to retrieve company's historical stock splits data.
    """
    ticker_obj = yf.Ticker(ticker)
    hist_splits = ticker_obj.get_splits()

    return hist_splits.to_dict()


@tool
def stock_news(ticker: str) -> dict:
    """
    Use this to retrieve latest news articles discussing particular stock ticker.
    """
    ticker_obj = yf.Ticker(ticker)

    return ticker_obj.get_news()


@tool
def company_retriever_tool(query: str) -> dict:
    """Use this tool to retrieve knowledge about Renault between 2020 and 2024 from the document database."""
    renault_retriever = get_retriever()
    response = get_response_with_sources(renault_retriever, query)

    return response["response"]


tools = [
    company_retriever_tool,
    company_information,
    last_dividend_and_earnings_date,
    stock_splits_history,
    summary_of_mutual_fund_holders,
    summary_of_institutional_holders,
    stock_grade_updrages_downgrades,
    stock_news,
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Try to answer user query using available tools."
            "If the question is about renault information between 2020 and 2024 start using retriever_tool with the entire query "
            "If you do not find information using retriever_tool, use another tool"
            "Do not use the same tool twice"
            "ticker:RNO.PA"
            "Do not mention the tools used."
        ),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


finance_agent = create_tool_calling_agent(model, tools, prompt)

finance_agent_executor = AgentExecutor(agent=finance_agent, tools=tools, verbose=True)
