.PHONY: run test test-unit test-int lint typecheck \
        migrate revision ci docker-build clean install mcp-config

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
	uv run lint-imports

typecheck:
	uv run mypy --strict src/

# ─── Database ─────────────────────────────────────────────────────────────────

migrate:
	docker compose run --rm app .venv/bin/alembic upgrade head

# Usage: make revision MESSAGE="add user table"
revision:
	docker compose run --rm app .venv/bin/alembic revision --autogenerate -m "$(MESSAGE)"

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

mcp-config:
	uv run python scripts/generate_mcp_config.py
