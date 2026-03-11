import fitz # PyMuPDF
import re
from typing import List, Tuple, Iterable


def extract_pages(path: str) -> List[Tuple[int, str]]:
    doc = fitz.open(path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        text = _strip_headers_footers(text)
        pages.append((i+1, text))
    doc.close()
    return pages


def _strip_headers_footers(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not re.match(r"^\s*Page\s+\d+\s*$", ln)]
    return "\n".join(lines)


def chunk_pages(pages: List[Tuple[int,str]], max_chars: int = 1800, overlap: int = 200) -> Iterable[Tuple[str, list]]:
    buf, pages_used = [], set()
    for pno, txt in pages:
        i = 0
    while i < len(txt):
        piece = txt[i:i+max_chars]
        if not buf:
             pages_used = set()
        buf.append(piece)
        pages_used.add(pno)
        joined = " ".join(buf).strip()
        if len(joined) >= max_chars:
            yield joined, sorted(pages_used)
            carry = joined[-overlap:]
            buf = [carry]
            pages_used = set([pno])
        i += max_chars
    if buf:
        yield " ".join(buf).strip(), sorted(pages_used)

def normalize_pages(pages):
    out = []
    for p in pages or []:
        if isinstance(p, int):
            out.append(p)
        elif isinstance(p, str):
            s = p.strip()
            if "-" in s:
                a, b = s.split("-", 1)
                try:
                    a, b = int(a), int(b)
                    if a <= b:
                        out.extend(range(a, b + 1))
                    else:
                        out.extend([a, b])  # weird order, keep both
                except ValueError:
                    # skip or log; string isn't a valid range
                    continue
            elif s.isdigit():
                out.append(int(s))
            # else: skip or log
    # dedupe + sort (optional)
    return sorted(set(out))