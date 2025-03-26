import streamlit as st
from retriever import get_retriever
from utils import parse_docs
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from config.settings import model

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.page_content
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """
    prompt_content = [{"type": "text", "text": prompt_template}]
    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )
    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )

def get_response_with_sources(retriever, question):
    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(RunnableLambda(build_prompt) | model | StrOutputParser())
    )
    return chain_with_sources.invoke(question)

def show_retriever_app(question):
    # Initialize retriever
    retriever = get_retriever()
    with st.spinner("Processing..."):
        response = get_response_with_sources(retriever, question)
    
        # Display the response
        st.write(response['response'])
        st.subheader("Context:")
        for text in response['context']['texts']:
            st.write("Source:", text.metadata.get('source', 'Unknown'))
            st.write("Chunk:", text.page_content)
            st.write("---")

        # Display images if any
        for i, image in enumerate(response['context']['images']):
            st.image(f"data:image/jpeg;base64,{image}", caption=f"Image {i+1}", use_container_width =True)

