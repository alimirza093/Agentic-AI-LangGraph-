from typing import TypedDict, Annotated

import streamlit as st

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# DARK THEME
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b0f14;
            color: #f5f7fa;
        }

        .main-title {
            text-align: center;
            font-size: 34px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .subtitle {
            text-align: center;
            color: #9ca3af;
            font-size: 15px;
            margin-bottom: 30px;
        }

        .programme-box {
            background-color: #111827;
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #1f2937;
            margin-bottom: 25px;
        }

        .chat-user {
            background-color: #1f2937;
            padding: 12px 16px;
            border-radius: 12px;
            margin: 8px 0;
        }

        .chat-assistant {
            background-color: #111827;
            padding: 12px 16px;
            border-radius: 12px;
            border: 1px solid #1f2937;
            margin: 8px 0;
        }

        .label {
            font-size: 12px;
            color: #9ca3af;
            margin-bottom: 5px;
            font-weight: 600;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODELS
# ============================================================

@st.cache_resource
def load_models():

    llm_model = ChatMistralAI(
        model="open-mistral-7b",
        temperature=0.4
    )

    embed_model = MistralAIEmbeddings(
        model="mistral-embed"
    )

    return llm_model, embed_model


llm_model, embed_model = load_models()


# ============================================================
# RETRIEVERS
# ============================================================

@st.cache_resource
def build_retrievers():

    def build_retriever(pdf_path: str):

        pdf = PyPDFLoader(pdf_path)
        docs = pdf.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embed_model
        )

        return vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

    academic_retriever = build_retriever(
        "01-LangGraph/01-Workflows/03-conditional-workflow/academics_handbook.pdf"
    )

    fee_retriever = build_retriever(
        "01-LangGraph/01-Workflows/03-conditional-workflow/fee_structure.pdf"
    )

    return academic_retriever, fee_retriever


academic_retriever, fee_retriever = build_retrievers()


# ============================================================
# STATE
# ============================================================

class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# ============================================================
# NODES
# ============================================================

def classifier_node(state: State) -> dict:

    last_message = state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"

        "Use 'academic' for questions about attendance, exams, grading, "
        "credits, promotion, course structure, summer training, or degree requirements.\n"

        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"

        "Use 'general' for greetings, casual talk, or anything not related "
        "to the college rules or fee.\n\n"

        f"Query: {last_message}\n\n"

        "Return only one word: academic, fee, or general."
    )

    response = llm_model.invoke(prompt)

    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"

    elif "fee" in category:
        category = "fee"

    else:
        category = "general"

    return {
        "query_type": category
    }


def academic_rag_node(state: State) -> dict:

    query = state["messages"][-1].content

    docs = academic_retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {
        "retrieved_context": context
    }


def fee_rag_node(state: State) -> dict:

    query = state["messages"][-1].content

    docs = fee_retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {
        "retrieved_context": context
    }


def general_node(state: State) -> dict:

    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


def response_node(state: State) -> dict:

    query = state["messages"][-1].content

    programme = state.get(
        "programme",
        "Unknown"
    )

    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = (
            f"You are a friendly college assistant talking to a "
            f"{programme} student. "

            f"Answer this question using your own general knowledge:\n\n"
            f"{query}"
        )

    else:

        prompt = (
            f"You are a college assistant helping a "
            f"{programme} student. "

            f"Use the following context from the official college "
            f"documents to answer the question accurately. "

            f"If the context mentions specific figures for different "
            f"programmes, highlight the one relevant to {programme} "
            f"if possible.\n\n"

            f"Context:\n{context}\n\n"

            f"Question: {query}\n\n"

            f"Give a clear, friendly and precise answer."
        )

    response = llm_model.invoke(prompt)

    return {
        "messages": [
            ("ai", response.content.strip())
        ]
    }


# ============================================================
# ROUTER
# ============================================================

def router(state: State):

    if state["query_type"] == "academic":
        return "academic"

    if state["query_type"] == "fee":
        return "fee"

    return "general"


# ============================================================
# GRAPH
# ============================================================

@st.cache_resource
def build_graph():

    graph = StateGraph(State)

    graph.add_node(
        "classifier",
        classifier_node
    )

    graph.add_node(
        "academic",
        academic_rag_node
    )

    graph.add_node(
        "fee",
        fee_rag_node
    )

    graph.add_node(
        "general",
        general_node
    )

    graph.add_node(
        "response",
        response_node
    )

    graph.add_edge(
        START,
        "classifier"
    )

    graph.add_conditional_edges(
        "classifier",
        router
    )

    graph.add_edge(
        "academic",
        "response"
    )

    graph.add_edge(
        "fee",
        "response"
    )

    graph.add_edge(
        "general",
        "response"
    )

    graph.add_edge(
        "response",
        END
    )

    return graph.compile()


app = build_graph()


# ============================================================
# UI
# ============================================================

st.markdown(
    '<div class="main-title">🎓 College Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Academic & Fee Information Assistant'
    '</div>',
    unsafe_allow_html=True
)


# Programme selection

st.markdown(
    '<div class="programme-box">',
    unsafe_allow_html=True
)

st.markdown("**Select your programme**")

programme = st.selectbox(
    "Programme",
    [
        "BCA",
        "BBA",
        "B.Com (H)"
    ],
    label_visibility="collapsed"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# Chat history

if "chat" not in st.session_state:
    st.session_state.chat = []


# Display previous messages

for message in st.session_state.chat:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="chat-user">
                <div class="label">YOU</div>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="chat-assistant">
                <div class="label">COLLEGE ASSISTANT</div>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )


# User input

user_query = st.chat_input(
    "Ask about academics, fees, exams, attendance..."
)


if user_query:

    # Show user message immediately

    st.session_state.chat.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    # Run LangGraph

    result = app.invoke(
        {
            "programme": programme,
            "messages": [
                ("human", user_query)
            ]
        }
    )

    answer = result["messages"][-1].content

    # Store assistant response

    st.session_state.chat.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()
