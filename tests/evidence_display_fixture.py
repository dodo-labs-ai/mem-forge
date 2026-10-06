"""Presentation for selector/transport fixtures, not semantic-accuracy evidence."""


def evidence_displays(primary_ref, required_refs=()):
    return [{"ref": ref, "text": f"Fixture source text for {ref}."}
            for ref in dict.fromkeys([primary_ref, *required_refs])]
