"""RAG evaluation runner — computes retrieval precision, recall, and faithfulness."""

from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import redis.asyncio as aioredis
import structlog

from src.ai.embeddings import EmbeddingService
from src.ai.eval.fixtures import (
    cleanup_eval_workspace,
    cleanup_stale_eval_workspaces,
    load_test_documents,
    setup_eval_workspace,
)
from src.ai.eval.metrics import (
    aggregate_precision,
    aggregate_recall,
    mean_reciprocal_rank,
    negative_rejection_rate,
    p95_latency,
    precision_at_k,
    recall,
    reciprocal_rank,
)
from src.ai.eval.models import EvalCase, EvalCaseResult, EvalMetrics, EvalResults
from src.core.cache import ResponseCache
from src.core.config import get_settings
from src.core.database import async_session_factory
from src.domain.roles import WorkspaceRole
from src.repositories.search import SQLAlchemySearchRepository
from src.services.search import SearchService

if TYPE_CHECKING:
    from src.services.ai import AIService

logger = structlog.get_logger(__name__)

_GOLDEN_DATASET_PATH = (
    Path(__file__).parent.parent.parent.parent / "evals" / "golden_dataset.json"
)
_TEST_DOCS_DIR = Path(__file__).parent.parent.parent.parent / "evals" / "test_documents"
_RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results"

_TOP_K = 5
_MIN_SCORE = 0.3


def _load_golden_dataset(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in data:
        missing = [
            f
            for f in ("id", "question", "category", "expected_answer")
            if f not in item
        ]
        if missing:
            raise ValueError(f"Eval case missing required fields {missing}: {item}")
        cases.append(
            EvalCase(
                id=item["id"],
                question=item["question"],
                expected_answer=item["expected_answer"],
                expected_source_doc_ids=item.get("expected_source_doc_ids", []),
                difficulty=item.get("difficulty", "medium"),
                category=item["category"],
                tags=item.get("tags", []),
            )
        )
    return cases


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


async def _judge_answer(
    question: str,
    answer: str,
    source_chunks: list[str],
) -> tuple[float, float]:
    """Run the judge agent; return (faithfulness, relevance) or (0.0, 0.0) on error."""
    from src.ai.agents.judge import judge_agent

    chunks_text = (
        "\n---\n".join(source_chunks) if source_chunks else "(no sources retrieved)"
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Source chunks used by the system:\n{chunks_text}\n\n"
        "Evaluate the faithfulness and relevance of the answer against the sources."
    )
    try:
        result = await judge_agent.run(prompt)
        return result.output.faithfulness, result.output.relevance
    except Exception:
        logger.warning("judge agent failed", question=question[:80], exc_info=True)
        return 0.0, 0.0


class EvalRunner:
    def __init__(
        self,
        search_service: SearchService,
        golden_dataset_path: Path,
        test_docs_dir: Path,
        embedding_service: EmbeddingService,
        ai_service: AIService | None = None,
    ) -> None:
        self._search_service = search_service
        self._golden_dataset_path = golden_dataset_path
        self._test_docs_dir = test_docs_dir
        self._embedding_service = embedding_service
        self._ai_service = ai_service
        self._workspace_id: uuid.UUID | None = None
        self._eval_user_id: uuid.UUID | None = None
        self._slug_to_id: dict[str, uuid.UUID] = {}

    async def setup(self) -> None:
        """Create the eval workspace and load all test documents with embeddings."""
        async with async_session_factory() as session:
            self._workspace_id, self._eval_user_id = await setup_eval_workspace(session)

        async with async_session_factory() as session:
            self._slug_to_id = await load_test_documents(
                workspace_id=self._workspace_id,
                docs_dir=self._test_docs_dir,
                session=session,
                embedding_service=self._embedding_service,
            )

        logger.info(
            "eval setup complete",
            workspace_id=str(self._workspace_id),
            documents=list(self._slug_to_id.keys()),
        )

    async def teardown(self) -> None:
        if self._workspace_id is None:
            return
        async with async_session_factory() as session:
            await cleanup_eval_workspace(self._workspace_id, session)
        self._workspace_id = None
        self._eval_user_id = None
        self._slug_to_id = {}

    def _uuid_to_slug(self, doc_id: uuid.UUID) -> str:
        for slug, uid in self._slug_to_id.items():
            if uid == doc_id:
                return slug
        return str(doc_id)

    async def _run_case(self, case: EvalCase) -> EvalCaseResult:
        assert self._workspace_id is not None

        t0 = time.monotonic()
        response = await self._search_service.search(
            workspace_id=self._workspace_id,
            query=case.question,
            top_k=_TOP_K,
            min_score=_MIN_SCORE,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        retrieved_slugs = [self._uuid_to_slug(r.document_id) for r in response.results]
        retrieved_chunks = [r.chunk_text for r in response.results]
        expected_slugs = case.expected_source_doc_ids

        p_at_k = precision_at_k(retrieved_slugs, expected_slugs, k=_TOP_K)
        r = recall(retrieved_slugs, expected_slugs)
        rr = reciprocal_rank(retrieved_slugs, expected_slugs)
        correctly_rejected = case.is_negative and len(response.results) == 0

        answer_text = ""
        answer_faithfulness = 0.0
        answer_relevance = 0.0
        negative_handling = False

        if self._ai_service is not None:
            assert self._eval_user_id is not None
            try:
                answer = await self._ai_service.ask(
                    workspace_id=self._workspace_id,
                    user_id=self._eval_user_id,
                    question=case.question,
                    role=WorkspaceRole.VIEWER,
                )
                answer_text = answer.answer
                faithfulness, relevance = await _judge_answer(
                    question=case.question,
                    answer=answer_text,
                    source_chunks=retrieved_chunks,
                )
                answer_faithfulness = faithfulness
                answer_relevance = relevance
                if case.is_negative:
                    negative_handling = answer.confidence <= 0.3
            except Exception:
                logger.warning("answer phase failed", case_id=case.id, exc_info=True)

        return EvalCaseResult(
            case_id=case.id,
            question=case.question,
            category=case.category,
            expected_source_doc_ids=expected_slugs,
            retrieved_doc_ids=retrieved_slugs,
            retrieved_chunks=retrieved_chunks,
            answer_text=answer_text,
            latency_ms=latency_ms,
            precision_at_k=p_at_k,
            recall=r,
            reciprocal_rank=rr,
            correctly_rejected=correctly_rejected,
            answer_faithfulness=answer_faithfulness,
            answer_relevance=answer_relevance,
            negative_handling=negative_handling,
        )

    async def run(self) -> EvalResults:
        cases = _load_golden_dataset(self._golden_dataset_path)
        logger.info("running eval", total_cases=len(cases))

        case_results: list[EvalCaseResult] = []
        for case in cases:
            result = await self._run_case(case)
            case_results.append(result)
            logger.debug(
                "case evaluated",
                case_id=case.id,
                precision=result.precision_at_k,
                recall=result.recall,
                latency_ms=round(result.latency_ms, 1),
            )

        positive_results = [r for r in case_results if r.category != "negative"]
        negative_results = [r for r in case_results if r.category == "negative"]

        faithfulness_scores = [
            r.answer_faithfulness for r in positive_results if r.answer_text
        ]
        relevance_scores = [
            r.answer_relevance for r in positive_results if r.answer_text
        ]
        neg_handling_flags = [
            r.negative_handling for r in negative_results if r.answer_text
        ]

        metrics = EvalMetrics(
            precision_at_k=aggregate_precision(
                [r.precision_at_k for r in positive_results]
            ),
            recall=aggregate_recall([r.recall for r in positive_results]),
            mrr=mean_reciprocal_rank([r.reciprocal_rank for r in positive_results]),
            negative_rejection_rate=negative_rejection_rate(
                [r.correctly_rejected for r in negative_results]
            ),
            p95_latency_ms=p95_latency([r.latency_ms for r in case_results]),
            total_cases=len(case_results),
            negative_cases=len(negative_results),
            positive_cases=len(positive_results),
            answer_faithfulness=(
                statistics.mean(faithfulness_scores) if faithfulness_scores else 0.0
            ),
            answer_relevance=(
                statistics.mean(relevance_scores) if relevance_scores else 0.0
            ),
            negative_handling_rate=(
                sum(neg_handling_flags) / len(neg_handling_flags)
                if neg_handling_flags
                else 0.0
            ),
        )

        return EvalResults(
            timestamp=datetime.now(UTC).isoformat(),
            git_commit=_git_commit(),
            metrics=metrics,
            cases=case_results,
        )


async def _main() -> None:
    cfg = get_settings()

    redis_client = aioredis.Redis.from_url(cfg.REDIS_URL, decode_responses=True)
    try:
        embedding_svc = EmbeddingService(
            api_key=cfg.EMBEDDING_API_KEY or cfg.GOOGLE_API_KEY or "",
            model=cfg.EMBEDDING_MODEL,
            dimensions=cfg.EMBEDDING_DIMENSIONS,
            redis_client=redis_client,
        )
        cache = ResponseCache(redis_client=redis_client)

        async with async_session_factory() as session:
            await cleanup_stale_eval_workspaces(session)

        async with async_session_factory() as session:
            from src.repositories.document import SQLAlchemyDocumentRepository
            from src.services.ai import AIService
            from src.services.document import DocumentService

            search_repo = SQLAlchemySearchRepository(session)
            search_svc = SearchService(
                search_repo=search_repo,
                embedding_service=embedding_svc,
                cache=cache,
            )
            doc_svc = DocumentService(
                repo=SQLAlchemyDocumentRepository(session),
                session=session,
            )
            ai_svc = AIService(
                search_service=search_svc,
                document_service=doc_svc,
            )

            runner = EvalRunner(
                search_service=search_svc,
                golden_dataset_path=_GOLDEN_DATASET_PATH,
                test_docs_dir=_TEST_DOCS_DIR,
                embedding_service=embedding_svc,
                ai_service=ai_svc,
            )

            try:
                await runner.setup()
                results = await runner.run()
            finally:
                await runner.teardown()

        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = _RESULTS_DIR / "current.json"
        output_path.write_text(
            json.dumps(results.to_dict(), indent=2),
            encoding="utf-8",
        )

        m = results.metrics
        logger.info(
            "eval complete",
            total_cases=m.total_cases,
            positive_cases=m.positive_cases,
            negative_cases=m.negative_cases,
            precision_at_k=round(m.precision_at_k, 3),
            recall=round(m.recall, 3),
            mrr=round(m.mrr, 3),
            negative_rejection_rate=round(m.negative_rejection_rate, 3),
            p95_latency_ms=round(m.p95_latency_ms),
            answer_faithfulness=round(m.answer_faithfulness, 3),
            answer_relevance=round(m.answer_relevance, 3),
            negative_handling_rate=round(m.negative_handling_rate, 3),
            output_path=str(output_path),
        )

    finally:
        await redis_client.aclose()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":  # pragma: no cover
    main()
