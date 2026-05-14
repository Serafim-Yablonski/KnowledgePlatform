from src.ai.chunking import ChunkingStrategy
from src.ai.chunking.markdown import MarkdownChunker
from src.ai.chunking.plaintext import PlainTextChunker
from src.domain.documents import ContentType


def get_chunker(content_type: ContentType) -> ChunkingStrategy:
    if content_type == ContentType.MARKDOWN:
        return MarkdownChunker()
    return PlainTextChunker()
