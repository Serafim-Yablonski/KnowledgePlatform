"""Unit tests for EvalRunner golden dataset loading and case classification."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ai.eval.models import EvalCase
from src.ai.eval.runner import EvalRunner, _load_golden_dataset

# ─── Golden dataset loading ───────────────────────────────────────────────────


class TestLoadGoldenDataset:
    def test_loads_valid_dataset(self, tmp_path: Path) -> None:
        data = [
            {
                "id": "eval_001",
                "question": "What is X?",
                "expected_answer": "X is Y",
                "expected_source_doc_ids": ["doc-a"],
                "difficulty": "easy",
                "category": "factual_lookup",
                "tags": ["test"],
            }
        ]
        f = tmp_path / "dataset.json"
        f.write_text(json.dumps(data))

        cases = _load_golden_dataset(f)
        assert len(cases) == 1
        assert cases[0].id == "eval_001"
        assert cases[0].question == "What is X?"
        assert cases[0].expected_source_doc_ids == ["doc-a"]
        assert cases[0].category == "factual_lookup"

    def test_rejects_case_missing_id(self, tmp_path: Path) -> None:
        data = [
            {"question": "What?", "expected_answer": "Y", "category": "factual_lookup"}
        ]
        f = tmp_path / "dataset.json"
        f.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="missing required fields"):
            _load_golden_dataset(f)

    def test_rejects_case_missing_question(self, tmp_path: Path) -> None:
        data = [{"id": "x", "expected_answer": "Y", "category": "factual_lookup"}]
        f = tmp_path / "dataset.json"
        f.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="missing required fields"):
            _load_golden_dataset(f)

    def test_rejects_case_missing_category(self, tmp_path: Path) -> None:
        data = [{"id": "x", "question": "Q?", "expected_answer": "A"}]
        f = tmp_path / "dataset.json"
        f.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="missing required fields"):
            _load_golden_dataset(f)

    def test_rejects_case_missing_expected_answer(self, tmp_path: Path) -> None:
        data = [{"id": "x", "question": "Q?", "category": "factual_lookup"}]
        f = tmp_path / "dataset.json"
        f.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="missing required fields"):
            _load_golden_dataset(f)

    def test_optional_fields_have_defaults(self, tmp_path: Path) -> None:
        data = [
            {
                "id": "eval_001",
                "question": "Q?",
                "expected_answer": "A",
                "category": "factual_lookup",
            }
        ]
        f = tmp_path / "dataset.json"
        f.write_text(json.dumps(data))

        cases = _load_golden_dataset(f)
        assert cases[0].difficulty == "medium"
        assert cases[0].tags == []
        assert cases[0].expected_source_doc_ids == []

    def test_multiple_cases_loaded(self, tmp_path: Path) -> None:
        data = [
            {
                "id": f"eval_{i:03d}",
                "question": f"Q{i}?",
                "expected_answer": f"A{i}",
                "category": "factual_lookup",
            }
            for i in range(10)
        ]
        f = tmp_path / "dataset.json"
        f.write_text(json.dumps(data))

        cases = _load_golden_dataset(f)
        assert len(cases) == 10


# ─── EvalCase.is_negative ─────────────────────────────────────────────────────


class TestEvalCaseIsNegative:
    def test_negative_category_is_negative(self) -> None:
        case = EvalCase(
            id="n1",
            question="Q?",
            expected_answer="No answer",
            expected_source_doc_ids=[],
            difficulty="easy",
            category="negative",
            tags=[],
        )
        assert case.is_negative is True

    def test_factual_lookup_is_not_negative(self) -> None:
        case = EvalCase(
            id="p1",
            question="Q?",
            expected_answer="A",
            expected_source_doc_ids=["doc-a"],
            difficulty="easy",
            category="factual_lookup",
            tags=[],
        )
        assert case.is_negative is False

    def test_multi_document_synthesis_is_not_negative(self) -> None:
        case = EvalCase(
            id="p2",
            question="Q?",
            expected_answer="A",
            expected_source_doc_ids=["doc-a", "doc-b"],
            difficulty="hard",
            category="multi_document_synthesis",
            tags=[],
        )
        assert case.is_negative is False

    def test_ambiguous_is_not_negative(self) -> None:
        case = EvalCase(
            id="p3",
            question="Q?",
            expected_answer="A",
            expected_source_doc_ids=["doc-a"],
            difficulty="medium",
            category="ambiguous",
            tags=[],
        )
        assert case.is_negative is False


# ─── EvalRunner._run_case logic ───────────────────────────────────────────────


class TestEvalRunnerCaseLogic:
    def _make_runner(
        self,
        search_results: list[dict],  # type: ignore[type-arg]
        slug_to_id: dict[str, uuid.UUID] | None = None,
    ) -> EvalRunner:
        from src.schemas.search import SearchResponse, SearchResultItem

        items = [
            SearchResultItem(
                chunk_text=r["chunk_text"],
                document_id=r["document_id"],
                document_title=r.get("title", "Doc"),
                score=r.get("score", 0.9),
            )
            for r in search_results
        ]
        response = SearchResponse(results=items, query="q", total_results=len(items))

        search_svc = MagicMock()
        search_svc.search = AsyncMock(return_value=response)

        embedding_svc = MagicMock()

        runner = EvalRunner(
            search_service=search_svc,
            golden_dataset_path=Path("/dev/null"),
            test_docs_dir=Path("/dev/null"),
            embedding_service=embedding_svc,
        )
        runner._workspace_id = uuid.uuid4()
        runner._slug_to_id = slug_to_id or {}
        return runner

    @pytest.mark.asyncio
    async def test_correctly_rejected_when_no_results_for_negative(self) -> None:
        runner = self._make_runner(search_results=[])
        case = EvalCase(
            id="n1",
            question="Q?",
            expected_answer="No answer",
            expected_source_doc_ids=[],
            difficulty="easy",
            category="negative",
            tags=[],
        )
        result = await runner._run_case(case)
        assert result.correctly_rejected is True
        assert result.precision_at_k == 0.0
        assert result.recall == 1.0  # vacuously true — no expected docs

    @pytest.mark.asyncio
    async def test_not_rejected_when_results_returned_for_negative(self) -> None:
        doc_id = uuid.uuid4()
        runner = self._make_runner(
            search_results=[{"chunk_text": "some text", "document_id": doc_id}]
        )
        case = EvalCase(
            id="n2",
            question="Q?",
            expected_answer="No answer",
            expected_source_doc_ids=[],
            difficulty="easy",
            category="negative",
            tags=[],
        )
        result = await runner._run_case(case)
        assert result.correctly_rejected is False

    @pytest.mark.asyncio
    async def test_precision_recall_for_positive_case(self) -> None:
        doc_a_id = uuid.uuid4()
        doc_b_id = uuid.uuid4()
        runner = self._make_runner(
            search_results=[
                {"chunk_text": "chunk a", "document_id": doc_a_id},
                {"chunk_text": "chunk b", "document_id": doc_b_id},
            ],
            slug_to_id={"doc-a": doc_a_id, "doc-b": doc_b_id},
        )
        case = EvalCase(
            id="p1",
            question="Q?",
            expected_answer="A",
            expected_source_doc_ids=["doc-a"],
            difficulty="easy",
            category="factual_lookup",
            tags=[],
        )
        result = await runner._run_case(case)
        assert result.precision_at_k == pytest.approx(0.5)  # 1 of 2 retrieved relevant
        assert result.recall == pytest.approx(1.0)  # doc-a was found
        assert result.reciprocal_rank == pytest.approx(1.0)  # doc-a is rank 1

    @pytest.mark.asyncio
    async def test_correctly_rejected_false_for_positive_case(self) -> None:
        doc_id = uuid.uuid4()
        runner = self._make_runner(
            search_results=[{"chunk_text": "text", "document_id": doc_id}],
            slug_to_id={"doc-a": doc_id},
        )
        case = EvalCase(
            id="p2",
            question="Q?",
            expected_answer="A",
            expected_source_doc_ids=["doc-a"],
            difficulty="easy",
            category="factual_lookup",
            tags=[],
        )
        result = await runner._run_case(case)
        # correctly_rejected is only meaningful for negative cases
        assert result.correctly_rejected is False
