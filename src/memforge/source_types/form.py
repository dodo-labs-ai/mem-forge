"""The shared form check: one reading of a source type's form description for every configuration.

A Gene's ``config_schema()`` declares its form: fields with conditions
(``visible_when``), labelled ``choices``, ``required`` meaning required while
visible, and booleans. The admin API checks every configuration against it
with :func:`check_config`, after the Gene's ``normalize_config`` and before
the type's own ``validate``. The admin UI mirrors the same reading, so the
form shows and hides exactly what the server checks.

Two rules make server and client agree:

- **Comparison string** (:func:`comparison_string`). A condition and the
  choices check compare values as strings: ``None``, ``""`` and an empty list
  are blank; a boolean is ``"true"`` or ``"false"``; a number is its decimal
  string; a string is stripped and lower-cased. A list, or a comma string of
  a ``tag_list`` or ``multi_select`` field, is its non-blank items joined by
  ``","``, so ``"A, B"`` and ``["a", "b"]`` compare equal.
- **Effective value** (:func:`effective_value`). A field reads the value in
  the configuration the save would store; if that is blank, the descriptor's
  ``scope_defaults`` entry; if that is still blank, the field's declared
  ``default``. Teams stores ``rolling_retention_days=None`` for "forever", so
  the stored value is blank and the field reads its declared default
  ``forever``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from memforge.genes.base import SourceConfigurationError
from memforge.genes.toolkit.config import FieldIssue, validate_bool_fields
from memforge.models import ConfigCondition, ConfigField, ConfigFieldType, ConfigOption, GeneConfigSchema
from memforge.source_types.config import secret_has_value
from memforge.source_types.descriptor import SourceTypeDescriptor

__all__ = [
    "check_config",
    "comparison_string",
    "conditions_hold",
    "effective_value",
    "visible_choices",
    "visible_fields",
]

_LIST_FIELD_TYPES = frozenset({ConfigFieldType.TAG_LIST, ConfigFieldType.MULTI_SELECT})


def comparison_string(value: Any, field_type: ConfigFieldType | None = None) -> str:
    """The string a condition or the choices check compares, ``""`` for a blank value::

        comparison_string(" PAT ")                                    # "pat"
        comparison_string(365)                                        # "365"
        comparison_string(True)                                       # "true"
        comparison_string(["a", " B", ""])                            # "a,b"
        comparison_string("A, B", ConfigFieldType.TAG_LIST)           # "a,b"
        comparison_string(None)                                       # ""
    """

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ",".join(item for item in (comparison_string(item) for item in value) if item)
    text = str(value)
    if field_type in _LIST_FIELD_TYPES:
        return comparison_string(text.split(","))
    return text.strip().lower()


def effective_value(field: ConfigField, config: Mapping[str, Any], descriptor: SourceTypeDescriptor) -> Any:
    """The stored value, else the descriptor's scope default, else the declared default."""

    value = config.get(field.key)
    if comparison_string(value, field.field_type):
        return value
    scope_default = descriptor.scope_defaults.get(field.key)
    if comparison_string(scope_default, field.field_type):
        return scope_default
    return field.default


def conditions_hold(
    conditions: Iterable[ConfigCondition],
    fields: Mapping[str, ConfigField],
    config: Mapping[str, Any],
    descriptor: SourceTypeDescriptor,
) -> bool:
    """Whether every condition holds for ``config``; ``fields`` maps each declared key to its field."""

    for condition in conditions:
        field = fields.get(condition.field)
        if field is None:
            return False
        value = comparison_string(effective_value(field, config, descriptor), field.field_type)
        if value not in {comparison_string(allowed) for allowed in condition.values}:
            return False
    return True


def visible_fields(
    schema: GeneConfigSchema,
    config: Mapping[str, Any],
    descriptor: SourceTypeDescriptor,
) -> list[ConfigField]:
    """The declared fields whose conditions hold for ``config``."""

    fields = {field.key: field for field in schema.fields}
    return [field for field in schema.fields if conditions_hold(field.visible_when, fields, config, descriptor)]


def visible_choices(
    field: ConfigField,
    fields: Mapping[str, ConfigField],
    config: Mapping[str, Any],
    descriptor: SourceTypeDescriptor,
) -> list[ConfigOption]:
    """The choices of ``field`` whose conditions hold for ``config``."""

    return [option for option in field.choices if conditions_hold(option.visible_when, fields, config, descriptor)]


def check_config(
    schema: GeneConfigSchema,
    config: Mapping[str, Any],
    existing_config: Mapping[str, Any],
    descriptor: SourceTypeDescriptor,
) -> list[FieldIssue]:
    """Every way ``config`` breaks its form description, in field order; empty when it passes.

    ``config`` is the configuration the save would store, after the Gene's
    ``normalize_config``. ``existing_config`` is the stored configuration,
    read only for whether a secret is already stored. The check refuses:

    - a visible required field without a value: a non-secret field reads its
      effective value, so a declared default satisfies it, and a secret reads
      the submitted or stored secret;
    - a visible ``tag_list`` or ``multi_select`` value that is neither a list
      nor a comma string;
    - a visible ``select`` value, or an item of a visible ``multi_select``
      value, that is not one of the field's visible choices;
    - any declared boolean, visible or not, that does not hold a boolean
      (:func:`~memforge.genes.toolkit.config.validate_bool_fields`), because
      the Gene checks every boolean when it is created.

    Hidden fields are otherwise ignored.
    """

    product_name = descriptor.gene_class().metadata().display_name
    fields = {field.key: field for field in schema.fields}
    issues: list[FieldIssue] = []
    for field in schema.fields:
        issue = _field_issue(field, fields, config, existing_config, descriptor, product_name)
        if issue is not None:
            issues.append(issue)
    return issues


def _issue_name(product_name: str, label: str) -> str:
    """Name a field after its product, sharing the words the two have in common.

    ``("GitHub Repository", "Repository URL")`` reads "GitHub Repository URL",
    and ``("Jira", "Jira Base URL")`` reads "Jira Base URL".
    """
    product_words, label_words = product_name.split(), label.split()
    for overlap in range(min(len(product_words), len(label_words)), 0, -1):
        if product_words[-overlap:] == label_words[:overlap]:
            return " ".join(product_words + label_words[overlap:])
    return f"{product_name} {label}"


def _field_issue(
    field: ConfigField,
    fields: Mapping[str, ConfigField],
    config: Mapping[str, Any],
    existing_config: Mapping[str, Any],
    descriptor: SourceTypeDescriptor,
    product_name: str,
) -> FieldIssue | None:
    if field.field_type == ConfigFieldType.BOOLEAN:
        try:
            validate_bool_fields(config, [field])
        except SourceConfigurationError as exc:
            return FieldIssue(field.key, f"{product_name} {exc}")
    if not conditions_hold(field.visible_when, fields, config, descriptor):
        return None
    value = config.get(field.key)
    name = _issue_name(product_name, field.label)
    if field.field_type in _LIST_FIELD_TYPES and value is not None and not isinstance(value, (list, str)):
        return FieldIssue(field.key, f"{name} must be a list or a comma-separated string")
    if field.required and not _has_value(field, config, existing_config, descriptor):
        return FieldIssue(field.key, f"{name} is required")
    if field.choices and field.field_type in {ConfigFieldType.SELECT, ConfigFieldType.MULTI_SELECT}:
        offered = visible_choices(field, fields, config, descriptor)
        allowed = {comparison_string(option.value) for option in offered}
        chosen = comparison_string(effective_value(field, config, descriptor), field.field_type)
        items = chosen.split(",") if field.field_type == ConfigFieldType.MULTI_SELECT else [chosen]
        if any(item and item not in allowed for item in items):
            labels = ", ".join(option.label for option in offered)
            return FieldIssue(field.key, f"{name} must be one of: {labels}")
    return None


def _has_value(
    field: ConfigField,
    config: Mapping[str, Any],
    existing_config: Mapping[str, Any],
    descriptor: SourceTypeDescriptor,
) -> bool:
    if field.field_type == ConfigFieldType.SECRET:
        return secret_has_value(field.key, config, existing_config)
    return bool(comparison_string(effective_value(field, config, descriptor), field.field_type))
