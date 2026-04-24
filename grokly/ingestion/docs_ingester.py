"""
grokly/ingestion/docs_ingester.py

Fetches ERPNext/Frappe documentation from URLs configured in
sources_docs.json and ingests text chunks into the ChromaStore.

URLs and crawl settings (delay, timeout, user_agent, min_text_length)
are all read from sources_docs.json via ConfigLoader.

Public API
----------
    from grokly.ingestion.docs_ingester import run
    chunks_added = run(store, config_loader)
"""

from __future__ import annotations

import hashlib
import time

import requests
from bs4 import BeautifulSoup

from grokly.store.chroma_store import ChromaStore

_CHUNK_SIZE    = 1200
_CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_module(url: str) -> str:
    """Derive a module label from the URL path."""
    if "/hr/" in url:
        return "hr"
    return url.split("/erpnext/")[-1].split("/")[0]


def _extract_content(soup: BeautifulSoup) -> str:
    """
    Remove noise tags then return text from the best content container.

    Search order: article → main → div.doc-content → div.content →
                  div.main → body
    """
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    container = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=lambda c: c and "doc-content" in c)
        or soup.find("div", class_=lambda c: c and "content" in c)
        or soup.find("div", class_=lambda c: c and "main" in c)
        or soup.body
    )
    return (container or soup).get_text(separator=" ", strip=True)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Character-based chunking with overlap. Drops empty chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - chunk_overlap
    return chunks


def _chunk_id(url: str, index: int) -> str:
    """md5(url + str(index)) → 32-char hex string."""
    return hashlib.md5((url + str(index)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(store: ChromaStore, config_loader) -> int:
    """
    Fetch every enabled URL from sources_docs.json, chunk the content, and
    add to *store*.

    Parameters
    ----------
    store : ChromaStore
        Destination knowledge base.
    config_loader : ConfigLoader
        Loaded configuration; supplies doc URLs and crawl settings.

    Returns
    -------
    int
        Total chunks added.
    """
    urls  = config_loader.get_enabled_doc_urls()
    crawl = config_loader.get_crawl_settings()

    headers  = {"User-Agent": crawl.get("user_agent", "Mozilla/5.0")}
    timeout  = crawl.get("timeout_seconds", 30)
    delay    = crawl.get("delay_seconds",   1.5)
    min_text = crawl.get("min_text_length", 300)

    total       = len(urls)
    fetched     = 0
    skipped     = 0
    total_added = 0

    for i, url in enumerate(urls, 1):
        print(f"  Fetching [{i}/{total}] {url}")

        try:
            response = requests.get(url, headers=headers, timeout=timeout)
        except requests.exceptions.ConnectionError as exc:
            print(f"  [ERROR] connection error — {exc}")
            skipped += 1
            time.sleep(delay)
            continue
        except requests.exceptions.Timeout:
            print(f"  [TIMEOUT] {url}")
            skipped += 1
            time.sleep(delay)
            continue
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            skipped += 1
            time.sleep(delay)
            continue

        if response.status_code != 200:
            print(f"  [SKIP] {url} — HTTP {response.status_code}")
            skipped += 1
            time.sleep(delay)
            continue

        try:
            soup = BeautifulSoup(response.text, "html.parser")
            text = _extract_content(soup)
        except Exception as exc:
            print(f"  [ERROR] parse failed — {exc}")
            skipped += 1
            time.sleep(delay)
            continue

        if len(text) < min_text:
            print(f"  [SKIP] {url} — only {len(text)} chars of text")
            skipped += 1
            time.sleep(delay)
            continue

        title_tag = soup.find("title")
        title     = title_tag.text.strip() if title_tag else url
        module    = _extract_module(url)

        chunks = _chunk_text(text, _CHUNK_SIZE, _CHUNK_OVERLAP)

        texts, metas, ids = [], [], []
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metas.append(
                {
                    "source":     "docs",
                    "url":        url,
                    "page_title": title[:200],
                    "module":     module,
                    "file_type":  "documentation",
                }
            )
            ids.append(_chunk_id(url, idx))

        added = store.upsert(texts, metas, ids)
        total_added += added
        fetched     += 1
        print(f"  [OK] {url} — {added} chunks added")

        time.sleep(delay)

    print(
        f"\n  Docs complete: {fetched} URLs fetched, "
        f"{skipped} skipped, {total_added} chunks total"
    )
    return total_added
