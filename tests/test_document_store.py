"""Behavioral contract tests for document artifact identity."""

from __future__ import annotations

from memforge.storage.document_store import LocalDocumentStore


def test_same_title_documents_have_distinct_local_artifacts(tmp_path) -> None:
    store = LocalDocumentStore(str(tmp_path / "documents"))

    first_raw = store.store_raw(
        source_id="src-repository",
        doc_id="github:org/repo:docs/user-guide.md",
        title="User Guide",
        content=b"first raw",
        content_type="text/markdown",
    )
    second_raw = store.store_raw(
        source_id="src-repository",
        doc_id="github:org/repo:examples/user-guide.md",
        title="User Guide",
        content=b"second raw",
        content_type="text/markdown",
    )
    first_normalized = store.store_normalized(
        source_id="src-repository",
        doc_id="github:org/repo:docs/user-guide.md",
        title="User Guide",
        markdown="# First",
    )
    second_normalized = store.store_normalized(
        source_id="src-repository",
        doc_id="github:org/repo:examples/user-guide.md",
        title="User Guide",
        markdown="# Second",
    )
    first_pdf = store.store_pdf(
        source_id="src-repository",
        doc_id="github:org/repo:docs/user-guide.md",
        title="User Guide",
        pdf_bytes=b"%PDF-first",
    )
    second_pdf = store.store_pdf(
        source_id="src-repository",
        doc_id="github:org/repo:examples/user-guide.md",
        title="User Guide",
        pdf_bytes=b"%PDF-second",
    )

    assert first_raw != second_raw
    assert first_normalized != second_normalized
    assert first_pdf != second_pdf
    assert store.read_artifact(first_raw) == b"first raw"
    assert store.read_artifact(second_raw) == b"second raw"
    assert store.read_artifact(first_normalized) == b"# First"
    assert store.read_artifact(second_normalized) == b"# Second"
    assert store.read_artifact(first_pdf) == b"%PDF-first"
    assert store.read_artifact(second_pdf) == b"%PDF-second"
