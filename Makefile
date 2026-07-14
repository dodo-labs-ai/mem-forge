.PHONY: install sync-plugin-mcp check-plugin-mcp lint test ui-install ui-lint ui-test ui-build check

install:
	uv sync --extra dev

sync-plugin-mcp:
	uv run python scripts/sync_plugin_mcp_proxy.py

check-plugin-mcp:
	uv run python scripts/sync_plugin_mcp_proxy.py --check

lint: check-plugin-mcp
	uv run ruff check src tests scripts/sync_plugin_mcp_proxy.py

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
