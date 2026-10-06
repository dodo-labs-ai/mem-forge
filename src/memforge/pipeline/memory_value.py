"""Source-neutral Value definition shared by extraction and candidate admission."""

MEMORY_VALUE_DEFINITION = """Worth remembering (keep): knowledge someone will still need later to act on or understand a
system, product or process, and that holds apart from the one event that produced it.
Examples: rules and requirements; designs and system behavior; decisions and their reasons;
conventions; causes of problems and how they are fixed; lasting ownership and
responsibilities; configuration and limits.
Not worth remembering (drop): a record of what happened once, which nobody needs after the
event. Examples: a single status transition; who an item was assigned to; a field changed
to some value; a version number bump; a link or parent/child relation between two items by
itself; who did what when; raw error text or log lines without a cause or conclusion;
scheduling and small talk.
Boundary: when an event establishes a lasting fact, the lasting fact is worth remembering (a
decision taken in a meeting is; an issue moving to Done is not). When unsure, keep.
Judge the knowledge a claim carries, not its tense or phrasing: a record stays a record
when it is phrased as a present fact. When a claim about a record also states a decision, a
reason or a requirement, judge it by that decision, reason or requirement. A claim that
states what a system, product, component or process is, does or requires is worth
remembering, whatever source it comes from."""
