from dataclasses import dataclass

@dataclass
class Document:
    text: str
    metadata: dict
    doc_id: str
