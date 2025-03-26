""" 
    UI for Renault Agent
"""

import streamlit as st
from langchain_core.messages import HumanMessage
from rag_app import show_retriever_app
from renault_agent import finance_agent_executor

st.set_page_config(page_title="Renault QA Agent", layout="wide")


st.title("📊 Renault QA Assistant")


# Session state to store the user question
if "question" not in st.session_state:
    st.session_state.question = ""


st.markdown("Welcome! Enter your question below to get started.")
st.markdown("The agent's knowledge base includes information on Renault's Renaulution strategy plan, based on our CEO Luca di Meo's talks, Renault's recent annual reports, Renault's stock prices for the current year, and the overall performance of the CAC40.")
question = st.text_input("Enter your question:", placeholder="e.g. Summarize the Renaultion plan report when it’s announced in 2021.?")

if question:
    with st.spinner("Processing..."):
        st.subheader("1/ Answer generation with Renault Agent")
        # Run agent
        response = finance_agent_executor.invoke(
            {"messages": [HumanMessage(content=question)]}
        )

        # Show Answer
        st.markdown(response.get("output", "No answer found."))
    with st.spinner("Processing..."):
        st.subheader("2/ Answer generation with RAG without agentic module")
        show_retriever_app(question)
