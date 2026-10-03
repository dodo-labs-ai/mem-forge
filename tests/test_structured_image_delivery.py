"""Public delivery mechanics; fake replies do not establish image understanding."""

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from pydantic import BaseModel
import pytest
from litellm.exceptions import ServiceUnavailableError
import json

from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig, StructuredLlmError
from memforge.llm.structured import _json_text_prompt, _structured_user_content
from memforge.llm.structured_image_delivery import ArtifactImageDelivery, DOCUMENTED_BOUNDS, digest, pixel_digest
from memforge.llm.structured_images import StructuredLlmImage, StructuredLlmImageError, prepare_structured_llm_images
from memforge.raster_views import complete_rectangles, native_fits
from tests.test_projection_fragments import BINARY_PROFILE, _compile, _projection

MODEL = "sap/anthropic--claude-4.6-sonnet"


class Reply(BaseModel):
    answer: str


def client(transport=None, **options):
    return LiteLlmStructuredClient(StructuredLlmConfig(
        MODEL, None, None, 60, native_schema_transport="json_schema_response_format",
        max_input_tokens=1_000_000, context_window_tokens=1_000_000, max_output_tokens=64_000, **options,
    ), completion_transport=transport)


def build(body, selected_client, **options):
    projection = _projection(context_profile=BINARY_PROFILE, context_content="", context_metadata={
        "source_artifact": {"inference_eligible": True, "sha256": sha256(body).hexdigest(),
                            "media_type": "image/png", "size_bytes": len(body), "filename": "fixture.png"},
    })
    catalog = _compile(projection, access_context_hash="fixture-access")
    artifact = next(f for f in catalog.fragments if f.kind.value == "artifact")
    return ArtifactImageDelivery.build(
        StructuredLlmImage(artifact.anchor.observation_id, "image/png", body), catalog=catalog,
        artifact_ref=artifact.reference, authority_hash="fixture-authority",
        model=MODEL, budget_identity=selected_client.request_budget(MODEL).identity, **options,
    )


@pytest.fixture(scope="module")
def authentic():
    body = (Path(__file__).parents[1] / "docs/design/images/revision-update-flow.png").read_bytes()
    assert sha256(body).hexdigest() == "1bddb850aeb0d2b51cf78a3a4c7649b2c422718bb51a504ea6e0f35ef16234aa"
    return build(body, client())


def test_actual_png_complete_frozen_source_pixels(authentic):
    assert len(authentic.views) == 7
    assert authentic.views[0].kind == "whole"
    assert authentic.views[0].body == prepare_structured_llm_images((authentic.original,)).images[0].body
    assert authentic.views[0].rectangle == (0, 0, 2040, 2410)
    assert authentic.identity()["capacity_evidence"] == "documented_only_route_unverified"
    covered = Image.new("1", (2040, 2410), 0)
    with Image.open(BytesIO(authentic.original.body)) as original:
        for view in authentic.views[1:]:
            assert native_fits(*view.dimensions, **DOCUMENTED_BOUNDS)
            with Image.open(BytesIO(view.body)) as wire, original.crop(view.rectangle) as source:
                assert pixel_digest(wire) == pixel_digest(source) == view.source_pixel_sha256
            covered.paste(1, view.rectangle)
    assert covered.getextrema() == (1, 1)
    assert prepare_structured_llm_images(authentic) is authentic
    assert len(complete_rectangles(2040, 2410, 19, **DOCUMENTED_BOUNDS, overlap_fraction=0.2)) == 6
    assert not native_fits(1122, 1326, **DOCUMENTED_BOUNDS)


@pytest.mark.parametrize("mutation", ["source", "wire", "order", "omit", "rectangle", "profile", "model", "budget"])
@pytest.mark.asyncio
async def test_invalid_bundle_never_calls_provider(authentic, mutation):
    calls = []

    async def transport(**kwargs):
        calls.append(kwargs)
        raise AssertionError("must reject before provider")

    changed = authentic
    if mutation == "source":
        changed = replace(changed, original=replace(changed.original, body=b"wrong-source"))
    elif mutation == "wire":
        views = (changed.views[0], replace(changed.views[1], body=b"wrong-view"), *changed.views[2:])
        changed = replace(changed, views=views)
        # Even a recomputed manifest cannot authorize pixels from another source.
        changed = replace(changed, manifest_sha256=digest(changed.identity()))
    elif mutation == "order":
        changed = replace(changed, views=tuple(reversed(changed.views)))
    elif mutation == "omit":
        changed = replace(changed, views=changed.views[:-1])
    elif mutation == "rectangle":
        changed = replace(changed, views=(changed.views[0], replace(changed.views[1], rectangle=(0, 0, 2, 2)),
                                          *changed.views[2:]))
    elif mutation == "profile":
        changed = replace(changed, profile="unknown")
    elif mutation == "model":
        changed = replace(changed, model="different-model")
    elif mutation == "budget":
        changed = replace(changed, budget_identity="different-budget")
    with pytest.raises((StructuredLlmImageError, StructuredLlmError)):
        await client(transport).evaluate_revision_work("same prompt", response_format=Reply,
                                                       max_tokens=32768, model=MODEL, images=changed)
    assert calls == []


@pytest.mark.asyncio
async def test_retry_native_fallback_preserves_exact_delivery_and_labels(authentic):
    calls = []

    async def transport(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ServiceUnavailableError("fixture transient", llm_provider="fixture", model=MODEL)
        content = '{"not_answer": true}' if len(calls) == 2 else '{"answer":"fake-only"}'
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))])

    selected = client(transport)
    assert selected.request_fits("same prompt", response_format=Reply, max_tokens=32768, images=authentic)
    # Compare actual text-only public count with delivery visual upper bound:
    # there is no automatic 85-token image contribution on the bundle path.
    tokens = selected.request_tokens("same prompt", response_format=Reply, images=authentic)
    assert tokens > authentic.visual_token_upper_bound == 7 * 1568
    result = await selected.evaluate_revision_work("same prompt", response_format=Reply, max_tokens=32768,
                                                   model=MODEL, images=authentic)
    assert result.answer == "fake-only" and len(calls) == 3
    image_sets = [[b for b in c["messages"][0]["content"] if b["type"] == "image_url"] for c in calls]
    assert image_sets[0] == image_sets[1] == image_sets[2] and len(image_sets[0]) == 7
    for call in calls:
        assert call["model"] == MODEL and call["max_tokens"] == 32768
        blocks = call["messages"][0]["content"]
        assert blocks[-1]["type"] == "text" and "same prompt" in blocks[-1]["text"]
        labels = [b["text"] for b in blocks[:-1] if b["type"] == "text"]
        assert len(labels) == 7 and "whole" in labels[0]
        assert all("not independent Evidence" in label and "authorized Artifact" in label for label in labels)
        assert all("Image evidence" not in label for label in labels)
    assert authentic.manifest_sha256 == digest(authentic.identity())


def test_small_odd_png_and_unsupported_orientation():
    buffer = BytesIO()
    Image.new("RGB", (31, 29)).save(buffer, format="PNG")
    small = build(buffer.getvalue(), client())
    assert len(small.views) == 1 and small.views[0].body == small.original.body
    for width, height in [(1101, 1399), (3333, 901)]:
        assert all(native_fits(right - left, bottom - top, **DOCUMENTED_BOUNDS) for left, top, right, bottom in
                   complete_rectangles(width, height, 19, **DOCUMENTED_BOUNDS, overlap_fraction=0.2))
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    Image.new("RGB", (31, 29)).save(buffer, format="PNG", exif=exif)
    with pytest.raises(StructuredLlmImageError):
        build(buffer.getvalue(), client())
    buffer = BytesIO()
    frame = Image.new("RGB", (31, 29), "red")
    frame.save(buffer, format="PNG", save_all=True, append_images=[Image.new("RGB", (31, 29), "blue")])
    with pytest.raises(StructuredLlmImageError):
        build(buffer.getvalue(), client())


def test_complete_capacity_failure_and_catalog_mismatch(authentic):
    with pytest.raises(StructuredLlmImageError):
        build(authentic.original.body, client(), max_images=1)
    limited = build(authentic.original.body, client(), max_encoded_request_bytes=1)
    with pytest.raises(StructuredLlmImageError):
        client().request_fits("same prompt", response_format=Reply, max_tokens=32768, images=limited)
    wrong = replace(authentic, source_binding=tuple((k, "another-observation" if k == "observation_id" else v)
                                                   for k, v in authentic.source_binding))
    with pytest.raises(StructuredLlmImageError):
        prepare_structured_llm_images(wrong)


@pytest.mark.asyncio
async def test_fallback_prompt_growth_rechecks_capacity_before_provider(authentic):
    calls = []

    async def transport(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                                                        message=SimpleNamespace(content='{"not_answer":true}'))])

    # Admit the initial fallback-size estimate exactly; extra actual fallback
    # diagnostics must not bypass that frozen declared request-byte boundary.
    encoded = len(json.dumps({"model": MODEL, "messages": [{"role": "user", "content": _structured_user_content(
        _json_text_prompt("same prompt", Reply), authentic)}], "schema": Reply.model_json_schema()},
        ensure_ascii=False).encode()) + 2048
    limited = replace(authentic, max_encoded_request_bytes=encoded)
    limited = replace(limited, manifest_sha256=digest(limited.identity()))
    selected = client(transport)
    assert selected.request_fits("same prompt", response_format=Reply, max_tokens=32768, images=limited)
    with pytest.raises(StructuredLlmError):
        await selected.evaluate_revision_work("same prompt", response_format=Reply, max_tokens=32768,
                                              images=limited)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_falsey_completion_transport_is_not_the_default_provider(monkeypatch):
    calls = []

    async def forbidden_default(**kwargs):
        raise AssertionError("explicit transport must not call the default provider")

    class FalseyTransport:
        def __bool__(self):
            return False

        async def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content='{"answer":"fake-only"}'),
            )])

    # Defensive provider-boundary interception keeps a regression offline.
    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", forbidden_default)
    response = await client(FalseyTransport()).evaluate_revision_work(
        "fixture prompt", response_format=Reply, max_tokens=32768, model=MODEL,
    )
    assert response.answer == "fake-only" and len(calls) == 1
