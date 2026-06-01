"""Unit tests for embed_chunks task helper functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


async def test_run_embedding_calls_embed_service() -> None:
    from src.workers.tasks.embed_chunks import _run_embedding

    mock_service = MagicMock()
    mock_service.embed_texts = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    with patch(
        "src.workers.tasks.embed_chunks.EmbeddingService",
        return_value=mock_service,
    ):
        result = await _run_embedding(
            texts=["hello world"],
            api_key="test-key",
            model="text-embedding-001",
            dimensions=3,
        )

    assert result == [[0.1, 0.2, 0.3]]
    mock_service.embed_texts.assert_awaited_once_with(["hello world"])
