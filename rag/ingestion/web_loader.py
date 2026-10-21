from trafilatura import fetch_url, extract, extract_metadata

def ingest_webpage(url: str):
    page = fetch_url(url)
    text = extract(page)
    meta = extract_metadata(page)

    data = [{
        "text": text,
        "metadata": {
            "source": url,
            "title": meta.title if meta else None,
            "type": "webpage"
        }
    }]

    return data
