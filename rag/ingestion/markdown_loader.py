import os
from mrkdwn_analysis import MarkdownAnalyzer

def ingest_md(filepath: str):
    analyzer = MarkdownAnalyzer(filepath)
    headers = analyzer.identify_headers()["Header"]

    with open(filepath, "r") as f:
        lines = f.readlines()

    data = []
    for i, header in enumerate(headers):
        start = header["line"]  # line after the heading
        end = headers[i + 1]["line"] - 1 if i + 1 < len(headers) else len(lines)

        text = "".join(lines[start:end]).strip()
        if text:
            data.append({
                "text": text,
                "metadata": {
                    "source": os.path.basename(filepath),
                    "section": header["text"],
                    "level": header["level"],
                    "type": "markdown"
                }
            })
    return data
