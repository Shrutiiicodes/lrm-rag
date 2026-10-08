"""Step 2 - Preprocessing & storage: chunk the text files, embed them, save a FAISS index."""
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA, STORE = Path("data"), "store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
YEAR = {1: "First Year", 2: "First Year", 3: "Second Year", 4: "Second Year",
        5: "Third Year", 6: "Third Year", 7: "Final Year", 8: "Final Year"}
ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th", 8: "8th"}
SCHEME_END = re.compile(r"Continuous Assessment \([AB]\)")  # (A) in Sem III-VIII PDFs, (B) in first year


def load_docs():
    # Faculty: one line per person is already a perfect chunk.
    docs = [Document(page_content=line, metadata={"source": "faculty"})
            for line in (DATA / "faculty.txt").read_text(encoding="utf-8").splitlines() if line.strip()]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    about = "[About the CSE (Data Science) department: history, establishment, intake]\n"
    docs += [Document(page_content=about + c, metadata={"source": "about"})
             for c in splitter.split_text((DATA / "about.txt").read_text(encoding="utf-8"))]
    for f in sorted(DATA.glob("syllabus_sem*.txt")):
        sem = int(f.stem.removeprefix("syllabus_sem"))
        # Prefix every chunk so "5th semester" style queries match chunks that never name the semester.
        prefix = f"[Semester {sem} ({ORD[sem]} semester, {YEAR[sem]}) CSE Data Science syllabus]\n"
        text = f.read_text(encoding="utf-8")
        # Each PDF opens with the scheme table (every course, elective and credit for the semester),
        # ending where the assessment rules start. Keep it whole: it answers "what subjects/electives".
        m = SCHEME_END.search(text)
        cut = m.start() if m else 0
        docs.append(Document(page_content=prefix + "Scheme - list of all courses, electives and credits:\n"
                             + text[:cut], metadata={"source": f.name, "semester": sem, "kind": "scheme"}))
        for chunk in splitter.split_text(text[cut:]):
            docs.append(Document(page_content=prefix + chunk,
                                 metadata={"source": f.name, "semester": sem, "kind": "detail"}))
    return docs


def embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


if __name__ == "__main__":
    docs = load_docs()
    FAISS.from_documents(docs, embeddings()).save_local(STORE)
    print(f"indexed {len(docs)} chunks -> {STORE}/")
