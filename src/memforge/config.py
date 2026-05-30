"""MemForge — Auto-evolutionary agent memory layer for development teams."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

__all__ = ["AppConfig", "load_config"]

DEFAULT_BASE_DIR = Path.home() / ".memforge"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StorageConfig:
    db_path: str = ""
    chroma_path: str = ""
    docs_path: str = ""

    def resolve(self, base_dir: Path) -> None:
        def resolve_path(value: str, default: Path) -> str:
            path = Path(value).expanduser() if value else default
            if not path.is_absolute():
                path = base_dir / path
            return str(path)

        if not self.db_path:
            self.db_path = str(base_dir / "db" / "memforge.db")
        if not self.chroma_path:
            self.chroma_path = str(base_dir / "vectors" / "chroma")
        if not self.docs_path:
            self.docs_path = str(base_dir / "documents")
        self.db_path = resolve_path(self.db_path, base_dir / "db" / "memforge.db")
        self.chroma_path = resolve_path(self.chroma_path, base_dir / "vectors" / "chroma")
        self.docs_path = resolve_path(self.docs_path, base_dir / "documents")


@dataclass
class LlmConfig:
    enrichment_model: str = "claude-sonnet-4-20250514"
    enrichment_base_url: str = "https://api.anthropic.com"
    enrichment_api_key: str = ""
    enrichment_max_tokens: int = 64000
    enrichment_max_concurrent: int = 3
    request_timeout_s: float = 300.0
    llm_calls_per_minute: int = 30
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = ""


@dataclass
class MemoryConfig:
    dedup_cosine_threshold: float = 0.08
    decay_schedule: str = "weekly"


@dataclass
class RetrievalConfig:
    default_top_k: int = 10
    rrf_k: int = 60
    recency_half_life_days: int = 90
    embedding_cache_size: int = 256
    enable_reranking: bool = False
    rerank_model: str = "claude-haiku-4-5-20251001"
    rerank_candidates: int = 30
    entity_model: str = "claude-haiku-4-5-20251001"
    entity_timeout_s: float = 1.0


@dataclass
class ServerConfig:
    admin_api_port: int = 8765
    jwt_secret: str = ""
    cors_origins: str = "*"  # comma-separated origins, or "*" for dev


@dataclass
class AppConfig:
    base_dir: Path = field(default_factory=lambda: DEFAULT_BASE_DIR)
    storage: StorageConfig = field(default_factory=StorageConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def __post_init__(self) -> None:
        if os.environ.get("MEMFORGE_BASE_DIR"):
            self.base_dir = Path(os.environ["MEMFORGE_BASE_DIR"]).expanduser()
        if os.environ.get("MEMFORGE_STORAGE_DB_PATH"):
            self.storage.db_path = os.environ["MEMFORGE_STORAGE_DB_PATH"]
        if os.environ.get("MEMFORGE_STORAGE_CHROMA_PATH"):
            self.storage.chroma_path = os.environ["MEMFORGE_STORAGE_CHROMA_PATH"]
        if os.environ.get("MEMFORGE_STORAGE_DOCS_PATH"):
            self.storage.docs_path = os.environ["MEMFORGE_STORAGE_DOCS_PATH"]
        self.storage.resolve(self.base_dir)
        # Environment variable overrides (MEMFORGE_* prefix)
        self.llm.enrichment_model = (
            os.environ.get("MEMFORGE_ENRICHMENT_MODEL")
            or self.llm.enrichment_model
        )
        self.llm.enrichment_base_url = (
            os.environ.get("MEMFORGE_ENRICHMENT_BASE_URL")
            or self.llm.enrichment_base_url
        )
        self.llm.enrichment_api_key = (
            os.environ.get("MEMFORGE_ENRICHMENT_API_KEY")
            or self.llm.enrichment_api_key
        )
        self.llm.request_timeout_s = float(
            os.environ.get("MEMFORGE_LLM_REQUEST_TIMEOUT_SECONDS")
            or self.llm.request_timeout_s
        )
        self.llm.embedding_model = (
            os.environ.get("MEMFORGE_EMBEDDING_MODEL")
            or self.llm.embedding_model
        )
        self.llm.embedding_base_url = (
            os.environ.get("MEMFORGE_EMBEDDING_BASE_URL")
            or self.llm.embedding_base_url
        )
        self.llm.embedding_api_key = (
            os.environ.get("MEMFORGE_EMBEDDING_API_KEY")
            or self.llm.embedding_api_key
        )
        self.server.admin_api_port = int(
            os.environ.get("MEMFORGE_ADMIN_API_PORT")
            or self.server.admin_api_port
        )
        self.server.cors_origins = (
            os.environ.get("MEMFORGE_CORS_ORIGINS")
            or self.server.cors_origins
        )
        self.server.jwt_secret = (
            os.environ.get("MEMFORGE_JWT_SECRET")
            or self.server.jwt_secret
            or "dev-secret-change-me"
        )


# ---------------------------------------------------------------------------
# TOML loader
# ---------------------------------------------------------------------------

def _set_nested(obj: object, keys: list[str], value: object) -> None:
    """Set a nested attribute on a dataclass from dotted TOML keys."""
    for key in keys[:-1]:
        obj = getattr(obj, key, None)
        if obj is None:
            return
    field_name = keys[-1]
    if hasattr(obj, field_name):
        current = getattr(obj, field_name)
        # Coerce types
        if isinstance(current, Path):
            value = Path(str(value))
        elif isinstance(current, int) and isinstance(value, (int, float)):
            value = int(value)
        elif isinstance(current, float) and isinstance(value, (int, float)):
            value = float(value)
        setattr(obj, field_name, value)


def load_config(config_path: Path | None = None, base_dir: Path | None = None) -> AppConfig:
    """Load configuration from TOML file with environment variable overrides.

    Priority: env vars > TOML file > defaults.
    """
    cfg = AppConfig(base_dir=base_dir or DEFAULT_BASE_DIR)

    if base_dir is not None:
        cfg.base_dir = base_dir

    # Try loading TOML
    toml_path = config_path or (cfg.base_dir / "config.toml")
    if toml_path.exists():
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        for section_name, section_data in data.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    _set_nested(cfg, [section_name, key], value)
            else:
                if hasattr(cfg, section_name):
                    setattr(cfg, section_name, section_data)

    # Re-resolve paths after TOML overrides
    cfg.storage.resolve(cfg.base_dir)
    # Re-apply env var overrides (they take priority over TOML)
    cfg.__post_init__()

    return cfg
