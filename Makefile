.PHONY: install lint test ui-install ui-lint ui-test ui-build check

install:
	uv sync --extra dev

lint:
	uv run ruff check src tests

test:
	uv run pytest -q

ui-install:
	cd admin-ui && npm ci

ui-lint:
	cd admin-ui && npm run lint

ui-test:
	cd admin-ui && npm test

ui-build:
	cd admin-ui && npm run build

check: lint test ui-lint ui-test ui-build
