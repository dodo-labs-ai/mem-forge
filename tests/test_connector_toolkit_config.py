"""The connector toolkit's configuration helpers."""

from __future__ import annotations

import ssl
from pathlib import Path

import pytest

from memforge.genes.base import SourceConfigurationError
from memforge.genes.toolkit.config import (
    bearer_headers,
    bool_config,
    int_config,
    list_config,
    require_https_base_url,
    resolve_pat,
    tls_ca_bundle,
    tls_verify,
    validate_bool_fields,
    validate_tls_ca_bundle,
)
from memforge.models import ConfigField, ConfigFieldType
from tests.tls_bundle_fixture import BUNDLE_CA_COUNT, trusted_ca_count, write_ca_bundle


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("false", False),
        ("yes", True),
        ("no", False),
        ("on", True),
        ("off", False),
        ("1", True),
        ("0", False),
        ("TRUE", True),
        ("False", False),
        ("Yes", True),
        ("NO", False),
        ("On", True),
        ("oFF", False),
        ("  true  ", True),
        ("\tno\n", False),
        (" 1 ", True),
    ],
)
def test_bool_config_reads_each_accepted_value(value, expected):
    assert bool_config({"flag": value}, "flag", default=not expected) is expected


@pytest.mark.parametrize("default", [True, False])
@pytest.mark.parametrize("config", [{}, {"flag": None}, {"flag": ""}, {"flag": "   "}])
def test_bool_config_reads_a_missing_or_empty_value_as_the_default(config, default):
    assert bool_config(config, "flag", default=default) is default


@pytest.mark.parametrize(
    "value",
    ["maybe", "2", "-1", "t", "f", "y", "n", "enabled", "none", "null", 2, -1, 1.0, 0.0, [], ["true"], {}, object()],
)
@pytest.mark.parametrize("default", [True, False])
def test_bool_config_refuses_any_other_value_naming_the_field(value, default):
    with pytest.raises(SourceConfigurationError, match="include_comments must be true or false"):
        bool_config({"include_comments": value}, "include_comments", default=default)


def test_validate_bool_fields_checks_only_boolean_fields():
    fields = [
        ConfigField(key="include_comments", label="Include Comments", field_type=ConfigFieldType.BOOLEAN),
        ConfigField(key="jql", label="JQL", field_type=ConfigFieldType.STRING),
    ]

    validate_bool_fields({"include_comments": "off", "jql": "maybe"}, fields)
    validate_bool_fields({}, fields)
    with pytest.raises(SourceConfigurationError, match="include_comments"):
        validate_bool_fields({"include_comments": "maybe"}, fields)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"max_files": 50}, 50),
        ({"max_files": "50"}, 50),
        ({"max_files": 0}, 1),
        ({"max_files": -3}, 1),
        ({"max_files": "lots"}, 500),
        ({"max_files": None}, 500),
        ({}, 500),
    ],
)
def test_int_config_reads_a_positive_integer_or_the_default(config, expected):
    assert int_config(config, "max_files", default=500) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (["PAY", " OPS ", "", "  "], ["PAY", "OPS"]),
        ("PAY, OPS,", ["PAY", "OPS"]),
        ("", []),
        (None, []),
        (42, []),
    ],
)
def test_list_config_reads_a_list_or_a_comma_separated_string(value, expected):
    assert list_config(value) == expected


def test_resolve_pat_prefers_the_configured_token(monkeypatch):
    monkeypatch.setenv("WIKI_TOKEN", "from-env")

    assert resolve_pat({"pat": " abc ", "pat_env_var": "WIKI_TOKEN"}, "Confluence") == "abc"
    assert resolve_pat({"pat_env_var": "WIKI_TOKEN"}, "Confluence") == "from-env"


def test_resolve_pat_refuses_a_missing_token(monkeypatch):
    monkeypatch.delenv("WIKI_TOKEN", raising=False)

    with pytest.raises(ValueError, match="Jira PAT is required"):
        resolve_pat({"pat_env_var": "WIKI_TOKEN"}, "Jira")
    with pytest.raises(ValueError, match="Jira PAT is required"):
        resolve_pat({}, "Jira")


def test_bearer_headers_send_the_token_and_ask_for_json():
    assert bearer_headers({"pat": "abc"}, "Confluence") == {
        "Authorization": "Bearer abc",
        "Accept": "application/json",
    }


def test_require_https_base_url():
    require_https_base_url("https://jira.example.test", "Jira")
    with pytest.raises(ValueError, match="Jira base_url must use HTTPS"):
        require_https_base_url("http://jira.example.test", "Jira")


def _ca_bundle(directory: Path) -> Path:
    return write_ca_bundle(directory / "corp-ca.pem")


def test_validate_tls_ca_bundle_refuses_a_missing_file(tmp_path):
    validate_tls_ca_bundle({})
    validate_tls_ca_bundle({"tls_ca_bundle": str(_ca_bundle(tmp_path))})
    with pytest.raises(ValueError, match="TLS CA bundle does not exist or is not readable"):
        validate_tls_ca_bundle({"tls_ca_bundle": str(tmp_path / "missing.pem")})


def test_tls_verify_uses_the_default_trust_store_without_a_bundle(monkeypatch):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/elsewhere/ca.pem")

    assert tls_ca_bundle({}) is None
    assert tls_verify({}) is True


def test_tls_verify_trusts_the_configured_bundle(tmp_path):
    bundle = _ca_bundle(tmp_path)

    assert tls_ca_bundle({"tls_ca_bundle": f" {bundle} "}) == str(bundle)
    verify = tls_verify({"tls_ca_bundle": str(bundle)})
    assert isinstance(verify, ssl.SSLContext)
    assert verify.verify_mode == ssl.CERT_REQUIRED
    assert trusted_ca_count(verify) == BUNDLE_CA_COUNT


def test_tls_verify_expands_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _ca_bundle(tmp_path)

    assert tls_ca_bundle({"tls_ca_bundle": "~/corp-ca.pem"}) == str(tmp_path / "corp-ca.pem")


def test_tls_verify_refuses_a_missing_bundle(tmp_path):
    with pytest.raises(ValueError, match="TLS CA bundle"):
        tls_verify({"tls_ca_bundle": str(tmp_path / "missing.pem")})


def test_requests_ca_bundle_is_honoured_only_when_asked_and_after_the_configured_bundle(tmp_path, monkeypatch):
    env_bundle = tmp_path / "env"
    env_bundle.mkdir()
    env_bundle = _ca_bundle(env_bundle)
    configured = _ca_bundle(tmp_path)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(env_bundle))

    assert tls_ca_bundle({}, honor_requests_ca_bundle=True) == str(env_bundle)
    assert tls_ca_bundle({"tls_ca_bundle": str(configured)}, honor_requests_ca_bundle=True) == str(configured)
    assert trusted_ca_count(tls_verify({}, honor_requests_ca_bundle=True)) == BUNDLE_CA_COUNT
    assert tls_verify({}) is True
