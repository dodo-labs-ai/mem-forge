"""Gene registry -- registration, factory, and discovery for gene plugins.

The registry is a simple ``dict[str, type[Gene]]`` mapping gene names to their
concrete classes.  Genes register themselves via the ``@register_gene``
decorator.  The sync orchestrator uses ``create_gene()`` to instantiate them
and ``list_available_genes()`` to populate the UI.

Built-in genes (confluence, jira) are imported lazily at module load time.
If a concrete gene module is not yet available the import is silently skipped
with a debug-level log message so the rest of the system keeps working.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from memforge.genes.base import Gene
from memforge.models import GeneMetadata

if TYPE_CHECKING:
    pass

__all__ = [
    "GENE_REGISTRY",
    "register_gene",
    "create_gene",
    "list_available_genes",
    "Gene",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENE_REGISTRY: dict[str, type[Gene]] = {}


def register_gene(gene_cls: type[Gene]) -> type[Gene]:
    """Class decorator that registers a gene in the global registry.

    Usage::

        @register_gene
        class ConfluenceGene(Gene):
            ...

    The gene's ``metadata().name`` is used as the registry key.

    Raises
    ------
    TypeError
        If *gene_cls* is not a subclass of ``Gene``.
    ValueError
        If a gene with the same name is already registered.
    """
    if not (isinstance(gene_cls, type) and issubclass(gene_cls, Gene)):
        raise TypeError(f"register_gene expects a Gene subclass, got {gene_cls!r}")

    meta = gene_cls.metadata()
    name = meta.name

    if name in GENE_REGISTRY:
        existing = GENE_REGISTRY[name]
        if existing is not gene_cls:
            raise ValueError(
                f"Gene name {name!r} is already registered by {existing.__qualname__}; "
                f"cannot register {gene_cls.__qualname__}"
            )
        # Idempotent re-registration of the same class is a no-op.
        return gene_cls

    GENE_REGISTRY[name] = gene_cls
    logger.info("Registered gene %r (%s)", name, gene_cls.__qualname__)
    return gene_cls


def create_gene(name: str, config: dict, source_id: str) -> Gene:
    """Instantiate a registered gene by name.

    Parameters
    ----------
    name:
        Gene name as declared in its ``GeneMetadata.name`` field
        (e.g. ``"confluence"``).
    config:
        User-provided configuration values for this source.
    source_id:
        Unique identifier for the configured source instance.

    Returns
    -------
    Gene
        A fully constructed (but not yet authenticated) gene instance.

    Raises
    ------
    KeyError
        If no gene with *name* is registered.
    """
    if name not in GENE_REGISTRY:
        available = ", ".join(sorted(GENE_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown gene {name!r}.  Available genes: {available}"
        )

    cls = GENE_REGISTRY[name]
    logger.debug("Creating gene %r (class=%s, source_id=%s)", name, cls.__qualname__, source_id)
    return cls(config=config, source_id=source_id)


def list_available_genes() -> list[GeneMetadata]:
    """Return metadata for every registered gene type.

    The list is sorted alphabetically by gene name for deterministic output.
    """
    return [
        cls.metadata()
        for _, cls in sorted(GENE_REGISTRY.items())
    ]


# ---------------------------------------------------------------------------
# Lazy import of built-in genes
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    """Import and register built-in gene modules."""
    builtin_imports: list[tuple[str, str, str]] = [
        # (gene_name, module_path, class_name)
        ("agent_session", "memforge.genes.agent_session_gene", "AgentSessionGene"),
        ("confluence", "memforge.genes.confluence_gene", "ConfluenceGene"),
        ("github_pages", "memforge.genes.github_pages_gene", "GitHubPagesGene"),
        ("jira", "memforge.genes.jira_gene", "JiraGene"),
        ("teams", "memforge.genes.teams_gene", "TeamsGene"),
    ]

    import importlib

    for gene_name, module_path, class_name in builtin_imports:
        try:
            mod = importlib.import_module(module_path)
            gene_cls = getattr(mod, class_name)
            if gene_name not in GENE_REGISTRY:
                GENE_REGISTRY[gene_name] = gene_cls
                logger.info("Registered built-in gene %r", gene_name)
        except ImportError:
            logger.debug("Built-in gene module %r not yet available -- skipping", module_path)
        except Exception:
            logger.warning("Failed to load built-in gene %r", module_path, exc_info=True)


_register_builtins()
