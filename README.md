# Experiment 8: RAG chatbot for DJSCE CSE (Data Science)

**Live app: https://lrm-rag.streamlit.app/**

LangChain RAG over the department's faculty page, About Us page and DJS23 syllabus PDFs (Sem I to VIII), with a Streamlit chat UI.

| Step (assignment task) | File | What it does |
|---|---|---|
| Data collection | `scrape.py` | Scrapes faculty (name, designation, joining date) from the [department page](https://www.djsce.ac.in/courses.php?course_id=5&sr_no=91). Also takes the department's About Us text (start year, intake). Downloads the latest-scheme (DJS23) syllabus PDFs: Sem III to VIII from the department page, and Sem I and II from the common first-year PDF on the [Basic Science & Humanities page](https://www.djsce.ac.in/courses.php?course_id=19&sr_no=212). Converts everything to `data/*.txt` |
| Preprocessing & storage | `ingest.py` | Chunks the text (one chunk per faculty member, the scheme table as one chunk per semester, `RecursiveCharacterTextSplitter` for the rest). Embeds with `all-MiniLM-L6-v2` and saves a FAISS index to `store/` |
| RAG pipeline | `rag.py` | LLM query router (rewrites the question, picks topic and semesters), retriever with metadata filters, prompt and Groq `openai/gpt-oss-120b`. Groq has retired its Llama chat models |
| Frontend | `app.py` | Streamlit chatbot that keeps chat history, so follow-up questions work |

## Run
```bash
pip install -r requirements.txt
python scrape.py
python ingest.py
```
Get a free key at https://console.groq.com and paste it into `.env` (`GROQ_API_KEY=...`), then:
```bash
python rag.py          # CLI self-check on the sample questions
python -m streamlit run app.py   # chatbot UI
```

## Deployment
Deployed on Streamlit Community Cloud: **https://lrm-rag.streamlit.app/**. It redeploys automatically on every push to `main`.

The built FAISS index (`store/`) is committed, so the cloud app doesn't re-scrape. The Groq key is stored in the app's Secrets, not in the repo (`.env` is gitignored).

To deploy your own copy:
1. The repo must be public, or Streamlit must be given access to private repos.
2. On https://share.streamlit.io, click **Create app** and enter the GitHub URL as `https://github.com/<user>/<repo>/blob/main/app.py`.
3. Under **Advanced settings**, set Python **3.12** (the CPU-only torch wheel in `requirements.txt` is built for 3.12) and add `GROQ_API_KEY = "..."` under **Secrets**. Click **Save**, then **Deploy**.

## Design notes
- **Query routing:** before retrieval, a small LLM call (`gpt-oss-20b`, structured output) rewrites the question as a standalone question, using recent chat turns so follow-ups like "what about the next semester?" work. It also labels the topic (faculty, department, syllabus or other) and the semesters asked about, so indirect questions ("who's in charge of the DS branch?", "electives in third year, second half") reach the right data.
- MiniLM embeddings can't tell "5th semester" from "6th semester", so the routed semesters become a metadata filter.
- Faculty questions get the whole roster (about 20 short lines), not just the top-k matches, so counting questions like "how many joined before 2025?" work.
- Each syllabus PDF starts with a scheme table that lists every course and elective. It is kept as one chunk and always retrieved for semester questions.
