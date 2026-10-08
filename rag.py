"""Step 3 - RAG pipeline: route the question, retrieve relevant chunks from FAISS, answer with an LLM (Groq)."""
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from ingest import STORE, embeddings

load_dotenv()  # GROQ_API_KEY from .env

PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant for the Computer Science and Engineering (Data Science) department "
    "of Dwarkadas J. Sanghvi College of Engineering (DJSCE).\n"
    "Answer the question using ONLY the context below. Questions may be indirect or informal: work out what is "
    "being asked, and count, compare dates or combine facts from the context when needed. "
    "If the answer really is not in the context, say you don't know.\n"
    "In syllabus scheme tables, course codes containing 'PE' (e.g. DJS23DCPE611, DJS23DPE301) are "
    "Programme/Department Electives (not core), 'OE' are Open Electives (their options may be labs, e.g. "
    "DJS23DOE311L DevOps Laboratory, and count as the listed open elective choices), 'PC' are core courses, 'MD' are "
    "multidisciplinary courses, and codes ending in 'L' or named 'Laboratory' are labs.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

ROUTER = ChatPromptTemplate.from_template(
    "You route questions for a chatbot about the CSE (Data Science) department of DJSCE. Its data: the faculty "
    "list (names, designations, joining dates), the department's About Us page (history, start year, intake) "
    "and the syllabus/scheme for semesters 1 to 8 (1st year = sem 1-2, 2nd year = 3-4, 3rd year = 5-6, "
    "final year = 7-8).\n"
    "Rewrite the latest question as a clear, standalone question, resolving references to the conversation "
    "(e.g. 'and the next one?', 'what about sem 6'), then classify it.\n\n"
    "Conversation so far:\n{history}\n\nLatest question: {question}"
)


class Route(BaseModel):
    question: str = Field(description="The latest question rewritten as a clear standalone question")
    topic: Literal["faculty", "department", "syllabus", "other"] = Field(
        description="faculty = teachers/HOD/staff; department = history, start year, intake, general info; "
                    "syllabus = subjects, courses, electives, labs, credits, exams; other = anything else")
    semesters: list[int] = Field(description="Semesters (1-8) the question is about; empty if none specific")


class RAG:
    def __init__(self, k=6):
        self.k = k
        self.db = FAISS.load_local(STORE, embeddings(), allow_dangerous_deserialization=True)  # our own index
        # Swap LLM here, e.g. ChatGoogleGenerativeAI(model="gemini-2.0-flash") from langchain-google-genai.
        self.chain = PROMPT | ChatGroq(model="openai/gpt-oss-120b", temperature=0) | StrOutputParser()
        self.router = ROUTER | ChatGroq(model="openai/gpt-oss-20b", temperature=0).with_structured_output(Route)

    def route(self, question, history=()):
        # Keep the last few turns, trimmed: enough to resolve "what about the next one?" cheaply.
        convo = "\n".join(f"{role}: {text[:300]}" for role, text in list(history)[-6:]) or "(none)"
        return self.router.invoke({"history": convo, "question": question})

    def by_source(self, query, source, k=200):
        return self.db.similarity_search(query, k=k, filter={"source": source}, fetch_k=2000)

    def retrieve(self, r):
        q, sems = r.question, [s for s in r.semesters if 1 <= s <= 8]
        if sems:
            # MiniLM can't tell "5th" from "6th" semester, so semesters become a metadata filter.
            # Always include their scheme tables: the chunks listing every course/elective.
            scheme = self.db.similarity_search(q, k=len(sems), filter={"semester": sems, "kind": "scheme"},
                                               fetch_k=2000)
            detail = self.db.similarity_search(q, k=max(self.k - len(sems), 3),
                                               filter={"semester": sems, "kind": "detail"}, fetch_k=2000)
            return sorted(scheme, key=lambda d: d.metadata["semester"]) + detail
        docs = self.db.similarity_search(q, k=self.k)
        if r.topic == "faculty":
            # Counting/listing questions ("how many joined before 2025") need every faculty line,
            # not the top-k nearest. The roster is ~20 short lines, so send all of it.
            docs = self.by_source(q, "faculty") + [d for d in docs if d.metadata.get("source") != "faculty"]
        elif r.topic == "department":
            docs = self.by_source(q, "about") + [d for d in docs if d.metadata.get("source") != "about"]
        return docs

    def ask(self, question, history=()):
        """history: earlier (role, text) turns, oldest first. Returns (answer, docs, route)."""
        r = self.route(question, history)
        docs = self.retrieve(r)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        return self.chain.invoke({"context": context, "question": r.question}), docs, r


if __name__ == "__main__":
    rag = RAG()
    checks = [  # (question, history, regex the answer must contain)
        ("Who is the HOD for CSE DS department?", [], r"Kriti Srivastava"),
        ("Who's in charge of the data science branch?", [], r"Kriti Srivastava"),
        ("When did the department start?", [], r"2020"),
        ("How old is the DS branch, roughly when was it set up?", [], r"2020"),
        ("How many teachers joined before 2025?", [], r"\b16\b"),
        ("Who is the most recent addition to the faculty?", [], r"2025"),
        ("What subjects are taught in the 5th semester?", [], r"Design and Analysis of Algorithms"),
        ("What can I pick as an elective in my third year, second half?", [], r"Time Series Analysis"),
        ("What are the subjects taught, semester wise, in 1st year?", [], r"Mathematics"),
        ("what about the semester after that?", [("user", "What subjects are taught in the 5th semester?"),
                                                ("assistant", "Machine Learning-II, Design and Analysis of "
                                                              "Algorithms, Intelligent Systems, ...")],
         r"Natural Language|Reinforcement"),
        ("What is the fee for the MBA program?", [], r"(?i)don.t know"),
    ]
    failed = 0
    for q, hist, want in checks:
        answer, docs, r = rag.ask(q, hist)
        flat = " ".join(answer.split())  # model may emit NBSPs
        ok = re.search(want, flat) is not None
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  Q: {q}\n      route: {r.topic} {r.semesters} | {r.question}\n"
              f"      A: {flat[:200]}\n")
    assert not failed, f"{failed} check(s) failed"
