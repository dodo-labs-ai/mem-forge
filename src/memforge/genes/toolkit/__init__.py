"""Shared building blocks for document connectors.

A connector reads its provider through these modules instead of carrying its
own copy, so every built-in connector behaves the same way. Import the part
you need from its module::

    from memforge.genes.toolkit.config import bool_config, tls_verify

- :mod:`memforge.genes.toolkit.config` reads configuration values: booleans,
  integers, lists, personal access tokens, TLS trust and HTTPS base URLs.
- :mod:`memforge.genes.toolkit.local_packages` reads the packages a device
  submitted for a Source, from the run's manifest or the Source's inbox.

The toolkit is transport and configuration only. It creates no business or
lifecycle state.
"""
