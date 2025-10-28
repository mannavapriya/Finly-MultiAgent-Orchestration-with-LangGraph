!pip install -q --upgrade langgraph langchain

!pip install -q google-generativeai==0.8.5 google-ai-generativelanguage==0.6.15

!pip install -q langchain-google-genai langgraph python-dotdev

!pip install -q langchain

!pip install -q -U langchain-tavily

from dotenv import load_dotenv
import os, getpass

# Load API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = getpass.getpass("GOOGLE_API_KEY")

import google.generativeai as genai
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

!pip install -U langchain-google-genai

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2
)

!pip install -q tavily-python langchain-community

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from typing import TypedDict, Annotated, List, Union
import json

os.environ["TAVILY_API_KEY"] = getpass.getpass("TAVILY_API_KEY")

!pip install -q -U langchain-tavily

from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=2)
tools = [tool]

tool.invoke("Stock market news summary")

news_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert finance news reporter. ONLY respond to finance news related questions.

IMPORTANT RULES:
- If asked about non-finance topics (travel, weather, math, general questions), politely decline and redirect to finance news
- Always provide complete, well-formatted finance news with specific details
- Include financial definitions, financial analysis, latest finance news, and practical tips

Use the ReAct approach:
1. THOUGHT: Analyze what finance information is needed
2. ACTION: Search for current information about financial definitions, financial analysis, latest finance news, and practical tips
3. OBSERVATION: Process the search results
4. Provide a comprehensive, formatted response

Available tools:
- TavilySearch: Search for current travel information

Format your financial updates with:
- Companies on the rise
- Summary of financial updates
- Practical tips and recommendations"""),
    MessagesPlaceholder(variable_name="messages"),
])

llm_with_tools = llm.bind_tools(tools)

finance_news_agent = news_prompt | llm_with_tools

!pip install -q serpapi

import serpapi

# Set up SERP API key
os.environ["SERPAPI_API_KEY"] = getpass.getpass("SERPAPI_API_KEY")

US_FIN_CODE_MAP = {
    # --- Stocks (Equities) ---
    "apple": "AAPL:NASDAQ",
    "microsoft": "MSFT:NASDAQ",
    "google": "GOOGL:NASDAQ",
    "alphabet": "GOOGL:NASDAQ",
    "amazon": "AMZN:NASDAQ",
    "meta": "META:NASDAQ",
    "facebook": "META:NASDAQ",
    "tesla": "TSLA:NASDAQ",
    "nvidia": "NVDA:NASDAQ",
    "netflix": "NFLX:NASDAQ",
    "intel": "INTC:NASDAQ",
    "amd": "AMD:NASDAQ",
    "salesforce": "CRM:NYSE",
    "ibm": "IBM:NYSE",

    # --- Indexes ---
    "s&p 500": "SPX:INDEX",
    "sp500": "SPX:INDEX",
    "nasdaq 100": "NDX:INDEX",
    "dow jones": "DJI:INDEX",
    "russell 2000": "RUT:INDEX",

    # --- ETFs / Mutual Funds ---
    "spy": "SPY:ETF",
    "qqq": "QQQ:ETF",
    "dia": "DIA:ETF",
    "vanguard s&p 500": "VOO:ETF",
    "vanguard total stock market": "VTI:ETF",
    "ishares core s&p 500": "IVV:ETF",
    "fidelity 500 index fund": "FXAIX:MUTUALFUND",

    # --- Currencies (Forex) ---
    "usd": "USD:FX",
    "us dollar": "USD:FX",
    "eur": "EUR:FX",
    "euro": "EUR:FX",
    "jpy": "JPY:FX",
    "japanese yen": "JPY:FX",
    "gbp": "GBP:FX",
    "british pound": "GBP:FX",
    "cad": "CAD:FX",
    "canadian dollar": "CAD:FX",

    # --- Futures ---
    "crude oil": "CL:NYMEX",
    "crude oil futures": "CL:NYMEX",
    "gold": "GC:COMEX",
    "gold futures": "GC:COMEX",
    "silver": "SI:COMEX",
    "natural gas": "NG:NYMEX",
    "s&p 500 futures": "ES:ECBOT",
    "nasdaq futures": "NQ:ECBOT",
    "dow futures": "YM:ECBOT"
}

def normalize_fin_code(name: str) -> str:
    """
    Convert US financial instrument names to standard trading codes.
    Handles stocks, indexes, ETFs/mutual funds, currencies, and futures.
    """
    if not name:
        return name
    name_lower = name.strip().lower()
    return US_FIN_CODE_MAP.get(name_lower, name.upper())

from pydantic import BaseModel
from typing import Optional

class FinancialInstrumentParams(BaseModel):
    instrument_name: str

from langchain_core.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=FinancialInstrumentParams)

from langchain_core.prompts import ChatPromptTemplate

fin_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a financial assistant. Extract the financial instrument (stock, index, mutual fund, currency, or futures)
name mentioned in the user's message. Return it as a JSON object matching this format:

{{
  "instrument_name": "..."
}}

If multiple instruments are mentioned, extract the first one only.
"""),
    ("human", "{user_query}")
])

fin_extraction_chain = fin_extraction_prompt | llm | parser

import os
import requests

def search_financial_instruments(query: str) -> str:
    """
    Enhanced Google Finance search using SerpApi.
    Returns a human-readable summary with top financial info + news.
    """
    try:
        # 1️⃣ Extract instrument name
        result = fin_extraction_chain.invoke({"user_query": query})
        instrument_name = result.instrument_name
        print("Extracted instrument:", instrument_name)

        # 2️⃣ Normalize code
        instrument_code = normalize_fin_code(instrument_name)
        print("Normalized code:", instrument_code)
        ticker_only = instrument_code.split(":")[0]  # remove exchange for matching

        # 3️⃣ SerpApi request
        params = {
            "engine": "google_finance",
            "q": instrument_code,
            "api_key": os.environ.get("SERPAPI_API_KEY")
        }
        response = requests.get("https://serpapi.com/search", params=params)
        if response.status_code != 200:
            return f"⚠️ SerpApi request failed: {response.status_code} - {response.text}"

        data = response.json()

        # 4️⃣ Try exact match in markets.us
        markets_us = data.get("markets", {}).get("us", [])
        stock_entry = next((m for m in markets_us if m.get("stock") == ticker_only), None)

        # 5️⃣ Fallback to summary
        summary = data.get("summary", {})
        if not stock_entry:
            stock_entry = summary

        # 6️⃣ Extract key fields
        price = stock_entry.get("price") or summary.get("price")
        currency = stock_entry.get("currency") or summary.get("currency", "USD")
        movement = stock_entry.get("price_movement") or summary.get("price_movement") or {}
        move_dir = movement.get("movement", "")
        move_value = movement.get("value")
        move_pct = movement.get("percentage")

        market_cap = summary.get("market_cap")
        pe_ratio = summary.get("pe_ratio")
        dividend_yield = summary.get("dividend_yield")
        prev_close = summary.get("previous_close")
        open_price = summary.get("open")
        day_range = summary.get("day_range")
        fifty_two_week_range = summary.get("fifty_two_week_range")
        volume = summary.get("volume")
        exchange = summary.get("exchange")
        gfin_url = data.get("search_metadata", {}).get("google_finance_url") or stock_entry.get("link")

        # 7️⃣ Collect news from multiple possible keys
        news_items = data.get("news") or data.get("news_results") or data.get("articles") or []
        top_news = [
          {"title": n.get("title") or n.get("snippet"), "link": n.get("link")}
          for n in news_items[:5]
          if (n.get("title") or n.get("snippet")) and n.get("link")
        ]


        # 8️⃣ Build human-readable summary
        lines = [
            f"📈 **{instrument_name} ({ticker_only}{':' + exchange if exchange else ''})**",
            f"💰 **Current Price:** {price} {currency}",
            f"📊 **Movement:** {move_dir} ({move_value} / {move_pct}%)" if move_value else "",
            f"🔗 **Google Finance:** {gfin_url}",
        ]

        for news in top_news:
            lines.append(f"📰 **Financial Reads:**  {news['title']} [Read more]({news['link']})")

        # Remove empty lines
        output = "\n".join([line for line in lines if line and line.strip()])
        return output or f"⚠️ No usable data found for **{ticker_only}**."

    except Exception as e:
        return f"❌ Finance details search failed: {str(e)}"

print(search_financial_instruments("Tesla"))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

finance_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a financial analysis expert. ONLY respond to queries about US financial instruments: stocks, indexes, mutual funds, currencies, or futures.

IMPORTANT RULES:
- If asked about non-financial topics, politely decline and redirect to finance discussions
- Always use the search_financial_instruments tool to get current Google Finance data via SERP API
- Provide detailed information about financial instruments including recent performance, Q1/Q2/Q3/Q4 results, revenue, EPS, dividends, price movement, and market trends
- Include practical insights for potential investments (buy/hold/caution)
- You CAN search and analyze results for different criteria like performance metrics, revenue, EPS, dividends, and market sentiment

Available tools:
- search_financial_instruments: Search for financial instruments using Google Finance engine via SERP API

When searching financial instruments, extract or ask for:
- Instrument type (stock, index, mutual fund, currency, futures)
- Company, fund name, or ticker symbol
- Relevant period (quarter, year)
- Specific performance metrics of interest (revenue, net income, EPS, dividends, market movement)

Present results with:
- Instrument name and ticker
- Instrument type
- Key financial highlights (revenue, net income, EPS, dividends)
- Recent market reaction (price movement, % change, analyst sentiment)
- Trend summary (growing, stable, declining)
- Practical investment insights (buy/hold/caution recommendation)"""),
    MessagesPlaceholder(variable_name="messages"),
])

llm_with_finance_tools = llm.bind_tools([search_financial_instruments])
financial_advisor_agent = finance_prompt | llm_with_finance_tools

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def create_finance_router():
    """Router for finance-related queries, directing to either news or instrument search agent"""

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing expert for a financial assistant system.

        Analyze the user's query and decide which specialist agent should handle it:

        - FIN_SEARCH_AGENT: Specific queries about financial instruments (stocks, indexes, mutual funds, currencies, futures), quarterly performance, revenue, EPS, dividends, price movement, or investment analysis
        - NEWS_AGENT: General financial news, market trends, announcements, press releases, macroeconomic updates

        Respond with ONLY one word: FIN_SEARCH_AGENT or NEWS_AGENT

        Examples:
        "What’s the Q2 revenue for Google?" → FIN_SEARCH_AGENT
        "Should I buy Tesla stock this quarter?" → FIN_SEARCH_AGENT
        "Latest market news" → NEWS_AGENT
        "Any updates on the S&P 500 today?" → FIN_SEARCH_AGENT
        "Federal Reserve interest rate announcement" → NEWS_AGENT
        "Top gainers in NASDAQ" → NEWS_AGENT
        "How did Microsoft perform last quarter?" → FIN_SEARCH_AGENT
        "Stock market news summary" → NEWS_AGENT
        "Upcoming earnings reports for tech companies" → FIN_SEARCH_AGENT
        "Economic news today" → NEWS_AGENT"""),
        ("user", "Query: {query}")
    ])

    router_chain = router_prompt | llm | StrOutputParser()

    def route_query(state):
        """Router function for LangGraph - decides which finance agent to call next"""

        # Get the latest user message
        user_message = state["messages"][-1].content
        print(f"🧭 Router analyzing: '{user_message[:50]}...'")

        try:
            # Get LLM routing decision
            decision = router_chain.invoke({"query": user_message}).strip().upper()

            # Map to agent node names
            agent_mapping = {
                "FIN_SEARCH_AGENT": "financial_advisor_agent",
                "NEWS_AGENT": "finance_news_agent"
            }

            next_agent = agent_mapping.get(decision, "financial_news_agent")
            print(f"🎯 Router decision: {decision} → {next_agent}")

            return next_agent

        except Exception as e:
            print(f"⚠️ Router error, defaulting to financial_news_agent: {e}")
            return "financial_news_agent"

    return route_query

router = create_finance_router()
print("✅ Finance Router created for LangGraph!")

from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import BaseMessage

class FinancePlannerState(TypedDict):
    """Simple state schema for travel multiagent system"""

    # Conversation history - persisted with checkpoint memory
    messages: Annotated[List[BaseMessage], operator.add]

    # Agent routing
    next_agent: Optional[str]

    # Current user query
    user_query: Optional[str]

from langchain_core.messages import HumanMessage, AIMessage

def finance_news_agent_node(state):
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}

    last_msg = messages[-1]
    query = last_msg if isinstance(last_msg, str) else last_msg.content

    # Step 1: Ask LLM if query is finance-related
    relevance_prompt = f"""
You are a finance assistant. Determine if the following query is finance-related:
stocks, markets, investments, economic updates.

Query: "{query}"

Answer YES if it is finance-related, NO if it is not.
"""

    # Pass a list of messages directly
    relevance_response = llm.invoke([HumanMessage(content=relevance_prompt)]).content.strip().upper()

    if relevance_response != "YES":
        final_response = AIMessage(content="I'm a finance news assistant. I can only provide finance news and updates. Please ask a finance-related question.")
        return {"messages": messages + [final_response]}

    # Step 2: Query the finance news tool only if relevant
    try:
        tool_result = tool.invoke({"query": query, "prompt": news_prompt})
        summary = f"### Finance News Summary for: {query}\n"
        for i, item in enumerate(tool_result.get("results", []), start=1):
            summary += (
                f"\n**{i}. {item.get('title', 'No title')}**\n"
                f"Link: {item.get('url', 'No URL')}\n"
                f"Highlights: {item.get('content', 'No content')}\n"
            )
    except Exception as e:
        summary = f"Search failed: {str(e)}"

    final_response = AIMessage(content=summary)
    return {"messages": messages + [final_response]}

state = {"messages": ["Finance Market today"]}
result = finance_news_agent_node(state)
print(result["messages"][-1].content)

def financial_advisor_agent_node(state: FinancePlannerState):
    """Financial Advisor agent node"""
    messages = state["messages"]
    user_query = messages[-1].content
    tool_result = search_financial_instruments(user_query)
    from langchain_core.messages import HumanMessage
    return {"messages": [HumanMessage(content=tool_result)]}

def router_node(state: FinancePlannerState):
    """Router node - determines which agent should handle the query"""
    user_message = state["messages"][-1].content
    next_agent = router(state)

    return {
        "next_agent": next_agent,
        "user_query": user_message
    }

def route_to_agent(state: FinancePlannerState):
    """Conditional edge function - routes to appropriate agent based on router decision"""

    next_agent = state.get("next_agent")

    if next_agent == "finance_news_agent":
        return "finance_news_agent"
    elif next_agent == "financial_advisor_agent":
        return "financial_advisor_agent"
    else:
        # Default fallback
        return "finance_news_agent"

from langgraph.graph import StateGraph, END
from typing import Literal
from langgraph.checkpoint.memory import InMemorySaver

workflow = StateGraph(FinancePlannerState)

# Add all nodes to the graph
workflow.add_node("router", router_node)
workflow.add_node("finance_news_agent", finance_news_agent_node)
workflow.add_node("financial_advisor_agent", financial_advisor_agent_node)

workflow.set_entry_point("router")

# Add conditional edge from router to appropriate agent
workflow.add_conditional_edges(
    "router",
    route_to_agent,
    {
        "finance_news_agent": "finance_news_agent",
        "financial_advisor_agent": "financial_advisor_agent",
    }
)

# Add edges from each agent back to END
workflow.add_edge("finance_news_agent", END)
workflow.add_edge("financial_advisor_agent", END)


checkpointer = InMemorySaver()

# Compile the graph
finance_planner = workflow.compile(checkpointer=checkpointer)

print("✅ Finance Planning Graph built successfully!")

from IPython.display import Image, display

# Generate and display the graph
graph_image = finance_planner.get_graph().draw_mermaid_png()
display(Image(graph_image))

from langchain_core.messages import HumanMessage

def multi_turn_chat():
    """Multi-turn conversation with checkpoint memory"""
    print("💬 Multi-Agent Travel Assistant (Multi-turn Mode)")
    print("=" * 50)

    # For multi-turn, you need a consistent thread/session ID
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("\n🧑 You: ")

        if user_input.lower() == 'quit':
            print("👋 Ending chat.")
            break

        print(f"\n📊 Processing query...")

        # Correctly call invoke() with config as a keyword argument
        result = finance_planner.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )

        # Extract the assistant's latest message
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            response = result["messages"][-1].content
        else:
            response = str(result)

        print(f"\n🤖 Assistant: {response}")
        print("-" * 50)


# Test multi-turn conversation
multi_turn_chat()