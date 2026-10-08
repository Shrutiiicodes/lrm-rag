"""Step 1 - Data collection: faculty page + latest-scheme syllabus PDFs -> plain text in data/."""
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

BASE = "https://www.djsce.ac.in/courses.php?course_id=5&sr_no="
FACULTY_URL, SYLLABUS_URL = BASE + "91", BASE + "87"
# First year is common to all branches; syllabus lives on the Basic Science & Humanities page (course_id=19).
FIRST_YEAR_URL = ("http://www.djsce.ac.in/docs/courses/First/"
                  "Final%20Modified%20DJS23%20Scheme%20and%20Syllabus%20wef%202025-26.pdf")
SCHEMES = ["DJS 23", "DJS 22"]  # preferred column first, fallback second
ROMAN = {"III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}
DATA = Path("data")


def soup(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"  # site is UTF-8 but doesn't say so; default guess garbles quotes/dashes
    return BeautifulSoup(r.text, "html.parser")


def scrape_about():
    # "About Us" paragraph + "Key Points" table (year of establishment, intake, duration).
    row = soup(SYLLABUS_URL).select_one("div.course_summary_outer").find_parent("div", class_="row")
    text = "\n".join(c.get_text(" ", strip=True) for c in row.find_all("div", recursive=False))
    (DATA / "about.txt").write_text(text, encoding="utf-8")
    print(f"about: {len(text)} chars")


def scrape_faculty():
    lines = []
    for card in soup(FACULTY_URL).select("div.shadow"):
        spans = [s.get_text(" ", strip=True) for s in card.find_all("span")]
        if len(spans) < 3 or not spans[2].startswith("Joined on"):
            continue
        name, desig, joined = spans[:3]
        # Site mixes "02.07.2007" and "01-09-2023"; write one unambiguous form so the LLM can compare dates.
        d = datetime.strptime(re.sub(r"\D", ".", joined.removeprefix("Joined on").strip()), "%d.%m.%Y")
        article = "an" if desig[:1] in "AEIOU" else "a"
        role = "the Head of the Department (HOD)" if "Head" in desig else f"{article} {desig}"
        lines.append(f"{name} is {role} of the Computer Science and Engineering (Data Science) "
                     f"department at DJSCE. Designation: {desig}. Joined on {d.day} {d:%B %Y}.")
    (DATA / "faculty.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"faculty: {len(lines)} members")


def clean_url(href):
    href = href.split("admin/", 1)[-1] if "admin/https" in href else href
    return quote(href, safe=":/%")  # encode spaces, keep existing %XX


def pick_syllabus_urls():
    """Return {semester: url} using the DJS 23 column (DJS 22 if that cell is empty)."""
    table = next(t for t in soup(SYLLABUS_URL).find_all("table") if "DJS 23" in t.get_text())
    rows = [tr.find_all("td") for tr in table.find_all("tr")]
    header = [td.get_text(" ", strip=True) for td in rows[0]]
    cols = [header.index(s) for s in SCHEMES]
    picked = {}
    for tds in rows[1:]:
        for c in cols:
            a = tds[c].find("a") if c < len(tds) else None
            m = re.search(r"Sem (\w+):", tds[c].get_text(" ", strip=True)) if a else None
            if m and m.group(1) in ROMAN:
                picked[ROMAN[m.group(1)]] = clean_url(a["href"])
                break
    return picked


def pdf_pages(url, name):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    pdf = DATA / "pdfs" / name
    pdf.write_bytes(r.content)
    return [re.sub(r"[ \t]+", " ", p.extract_text() or "") for p in PdfReader(pdf).pages]


def save_sem(sem, text, url):
    (DATA / f"syllabus_sem{sem}.txt").write_text(text, encoding="utf-8")
    print(f"sem {sem}: {len(text):>7} chars  <- {url}")
    if len(text.strip()) < 500:
        print(f"  WARNING: sem {sem} PDF has almost no text (scanned?)")


def scrape_first_year():
    """One PDF covers Sem I + II: page 1 = Sem I scheme, page 2 = Sem II scheme, pages 3-4 = credit
    summary + assessment rules, then the Sem I detailed syllabus, then the Sem II one."""
    p = pdf_pages(FIRST_YEAR_URL, "sem1_2.pdf")
    s2 = next(i for i in range(5, len(p)) if "(Semester II)" in p[i])
    common = p[3] + p[4]  # starts with the assessment rules ingest uses as the scheme cut
    # Facts stated in the PDF but spread across pages; put them in the scheme chunk the LLM always sees.
    note = ("Note: First year is common to all programmes. Courses are split into Group A and Group B; "
            "a learner offered Group A in Sem I takes Group B in Sem II and vice versa. "
            "Each programme takes one Program Core (DJS23FPC15x) course in Sem II; for Computer Science "
            "and Engineering (Data Science) it is Fundamentals of Data Analysis (DJS23FPC155).\n")
    save_sem(1, "\n".join([note + p[1], common, *p[5:s2]]), FIRST_YEAR_URL)
    save_sem(2, "\n".join([note + p[2], common, *p[s2:]]), FIRST_YEAR_URL)


def scrape_syllabus():
    (DATA / "pdfs").mkdir(parents=True, exist_ok=True)
    scrape_first_year()
    for sem, url in sorted(pick_syllabus_urls().items()):
        save_sem(sem, "\n".join(pdf_pages(url, f"sem{sem}.pdf")), url)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    scrape_faculty()
    scrape_about()
    scrape_syllabus()
