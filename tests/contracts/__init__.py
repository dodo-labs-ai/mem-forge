"""Adapter contract test suite.

The modules here define behavior every storage adapter must satisfy, expressed
once and parameterized by an adapter factory. A new adapter wires itself into
the suite by subclassing the abstract base test classes and overriding the
``adapters`` fixture; the test bodies, builders, and assertions are reused
verbatim. The concrete SQLite/FTS/Chroma wiring lives in
``test_sqlite_contract.py`` and is the single in-repo consumer of the suite
today.
"""
