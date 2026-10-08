"""Step 4 - Frontend: Streamlit chatbot over the RAG pipeline. Run: python -m streamlit run app.py"""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # GROQ_API_KEY from .env
st.set_page_config(page_title="DJSCE CSE (DS) Assistant", page_icon="🎓")
st.title("🎓 DJSCE CSE (Data Science) Assistant")
st.caption("Answers from the department's faculty page, About Us page and DJS23 syllabus PDFs (Sem I to VIII).")

if not os.environ.get("GROQ_API_KEY"):
    # Streamlit Cloud exposes root-level Secrets as environment variables, so one check covers both.
    st.error("GROQ_API_KEY is not set. Locally: add it to `.env` and restart. "
             "On Streamlit Cloud: App settings → Secrets → `GROQ_API_KEY = \"...\"`.")
    st.stop()


@st.cache_resource(show_spinner="Loading vector store and model...")
def load_rag():
    from rag import RAG
    return RAG()


rag = load_rag()
if "history" not in st.session_state:
    st.session_state.history = []  # (role, text), oldest first; also lets follow-up questions resolve

for role, text in st.session_state.history:
    st.chat_message(role).markdown(text)

if question := st.chat_input("e.g. Who is the HOD? / What subjects are taught in the 5th semester?"):
    st.chat_message("user").markdown(question)
    with st.spinner("Searching and generating..."):
        try:
            answer, _, _ = rag.ask(question, st.session_state.history)
        except Exception as e:  # e.g. Groq free-tier rate limit: tell the user instead of a traceback
            st.chat_message("assistant").error(f"The language model request failed, please try again "
                                               f"in a minute. ({type(e).__name__})")
            st.stop()
    st.chat_message("assistant").markdown(answer)
    st.session_state.history += [("user", question), ("assistant", answer)]
