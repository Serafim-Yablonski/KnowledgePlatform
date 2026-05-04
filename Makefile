.PHONY: run test test-unit test-int lint typecheck eval eval-compare \
        migrate revision ci docker-build clean seed install

# ─── Development ──────────────────────────────────────────────────────────────

install:
	uv sync --extra dev

run:
	docker compose up

# ─── Testing ──────────────────────────────────────────────────────────────────

test:
	uv run pytest --cov=src --cov-fail-under=85

test-unit:
	uv run pytest tests/unit -x; \
	ret=$$?; if [ $$ret -eq 5 ]; then exit 0; fi; exit $$ret

test-int:
	uv run pytest tests/integration -x; \
	ret=$$?; if [ $$ret -eq 5 ]; then exit 0; fi; exit $$ret

# ─── Code Quality ─────────────────────────────────────────────────────────────

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

typecheck:
	uv run mypy --strict src/

# ─── AI Evaluation ────────────────────────────────────────────────────────────

eval:
	uv run python -m src.ai.eval.runner

eval-compare:
	uv run python -m src.ai.eval.compare results/baseline.json results/current.json

# ─── Database ─────────────────────────────────────────────────────────────────

migrate:
	uv run alembic upgrade head

# Usage: make revision MESSAGE="add user table"
revision:
	uv run alembic revision --autogenerate -m "$(MESSAGE)"

# ─── CI (mirrors GitHub Actions pipeline) ────────────────────────────────────

ci:
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

# ─── Docker ───────────────────────────────────────────────────────────────────

docker-build:
	docker build -t knowledge-platform .

# ─── Housekeeping ─────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true

seed:
	@echo "not implemented yet"
