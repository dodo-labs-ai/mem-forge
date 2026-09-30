.PHONY: install sync-plugin-mcp check-plugin-mcp check-admin-api lint test ui-install ui-lint ui-test ui-build check

install:
	uv sync --extra dev

sync-plugin-mcp:
	uv run python scripts/sync_plugin_mcp_proxy.py

check-plugin-mcp:
	uv run python scripts/sync_plugin_mcp_proxy.py --check

check-admin-api:
	uv run python scripts/export_openapi.py admin/openapi/admin.json --check

lint: check-plugin-mcp check-admin-api
	uv run ruff check src tests scripts/sync_plugin_mcp_proxy.py scripts/export_openapi.py

test:
	uv run pytest -q

# Deprecated (admin-ui-v1): each ui-* target also runs the V1 admin UI until
# it is removed (ADR 0044).
ui-install:
	cd admin && npm ci
	cd admin-ui && npm ci

ui-lint:
	cd admin && npm run lint && npm run check:api
	cd admin-ui && npm run lint

ui-test:
	cd admin && npm test
	cd admin-ui && npm test

ui-build:
	cd admin && npm run build
	cd admin-ui && npm run build

check: lint test ui-lint ui-test ui-build
