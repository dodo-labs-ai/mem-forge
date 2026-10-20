"""Public delivery mechanics; fake replies do not establish image understanding."""

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

from PIL import Image, ImageDraw
from pydantic import BaseModel
import pytest
from litellm.exceptions import ServiceUnavailableError
import json

from memforge.llm.batch_runner import _failure_from_error
from memforge.llm.structured import (
    INPUT_CAPACITY_EXCEEDED,
    LiteLlmStructuredClient,
    StructuredLlmConfig,
    StructuredLlmError,
)
from memforge.llm.structured import _json_text_prompt, _structured_user_content
from memforge.llm.structured_image_delivery import ArtifactImageDelivery, DOCUMENTED_BOUNDS, digest, pixel_digest
from memforge.llm.structured_images import StructuredLlmImage, StructuredLlmImageError, prepare_structured_llm_images
from memforge.raster_views import complete_rectangles, native_fits
from tests.test_projection_fragments import BINARY_PROFILE, _compile, _projection

MODEL = "litellm_proxy/claude-sonnet"


class Reply(BaseModel):
    answer: str


def client(transport=None, **options):
    limits = {"max_input_tokens": 1_000_000, "context_window_tokens": 1_000_000, "max_output_tokens": 64_000}
    return LiteLlmStructuredClient(StructuredLlmConfig(
        MODEL, None, None, 60, native_schema_transport="json_schema_response_format", **{**limits, **options},
    ), completion_transport=transport)


def edge_client(input_tokens, transport=None):
    """A client whose route holds exactly ``input_tokens`` of input when no correction room is kept."""
    return client(transport, max_input_tokens=input_tokens, input_budget_fraction=1.0)


def answering(calls, *replies):
    """A transport that answers with ``replies`` in turn, then with a valid reply."""
    async def transport(**kwargs):
        calls.append(kwargs)
        content = replies[len(calls) - 1] if len(calls) <= len(replies) else '{"answer":"yes"}'
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                                                        message=SimpleNamespace(content=content))])
    return transport


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


def tall_png():
    """A PNG larger than one native view, with distinct pixels in every tile."""
    image = Image.new("RGB", (2040, 2410))
    draw = ImageDraw.Draw(image)
    for top in range(0, 2410, 10):
        for left in range(0, 2040, 10):
            draw.rectangle((left, top, left + 9, top + 9), fill=(left % 256, top % 256, (left + top) % 256))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def authentic():
    return build(tall_png(), client())


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
        await client(transport).evaluate_revision_work("same prompt", response_format=Reply, model=MODEL, images=changed)
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
    assert selected.request_fits("same prompt", response_format=Reply, images=authentic)
    # Compare actual text-only public count with delivery visual upper bound:
    # there is no automatic 85-token image contribution on the bundle path.
    tokens = selected.request_tokens("same prompt", response_format=Reply, images=authentic)
    assert tokens > authentic.visual_token_upper_bound == 7 * 1568
    result = await selected.evaluate_revision_work("same prompt", response_format=Reply,
                                                   model=MODEL, images=authentic)
    assert result.answer == "fake-only" and len(calls) == 3
    image_sets = [[b for b in c["messages"][0]["content"] if b["type"] == "image_url"] for c in calls]
    assert image_sets[0] == image_sets[1] == image_sets[2] and len(image_sets[0]) == 7
    for call in calls:
        assert call["model"] == MODEL and call["max_tokens"] == selected.request_budget().output_tokens
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
    # A request over the encoded size limit does not fit, as one over the token budget does not.
    assert not client().request_fits("same prompt", response_format=Reply, images=limited)
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
    assert selected.request_fits("same prompt", response_format=Reply, images=limited)
    with pytest.raises(StructuredLlmError):
        await selected.evaluate_revision_work("same prompt", response_format=Reply,
                                              images=limited)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_request_that_fits_only_without_correction_room_is_sent(authentic):
    """A re-ask is packed without room for another correction, so the client sends what fits as it is."""
    tokens = client().request_tokens("same prompt", response_format=Reply, images=authentic)
    calls = []
    edge = edge_client(tokens, answering(calls))
    images = build(authentic.original.body, edge)

    assert not edge.request_fits("same prompt", response_format=Reply, images=images)
    assert edge.request_fits("same prompt", response_format=Reply, images=images, reserve_correction=False)
    reply = await edge.evaluate_revision_work("same prompt", response_format=Reply, images=images)

    assert reply.answer == "yes"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_request_beyond_the_route_is_a_capacity_failure_of_its_items(authentic):
    tokens = client().request_tokens("same prompt", response_format=Reply, images=authentic)
    calls = []
    edge = edge_client(tokens - 2, answering(calls))
    images = build(authentic.original.body, edge)

    with pytest.raises(StructuredLlmError) as raised:
        await edge.evaluate_revision_work("same prompt", response_format=Reply, images=images)

    assert raised.value.error_code == INPUT_CAPACITY_EXCEEDED
    assert raised.value.terminal_category == "request_error"
    assert _failure_from_error(raised.value).category == "capacity_exceeded"
    assert calls == []


@pytest.mark.asyncio
async def test_a_repair_request_is_counted_as_it_is_sent(authentic):
    """The repair of an invalid answer fits where the request, its schema instruction and its note fit, once each."""
    roomy_calls = []
    roomy = client(answering(roomy_calls, '{"not_answer":true}'))
    await roomy.evaluate_revision_work("same prompt", response_format=Reply, images=authentic)
    repair = roomy_calls[1]["messages"][0]["content"][0]["text"]
    sent = roomy._rendered_request_tokens(repair, response_format=Reply, model=MODEL, images=authentic)
    calls = []
    edge = edge_client(sent, answering(calls, '{"not_answer":true}'))

    reply = await edge.evaluate_revision_work(
        "same prompt", response_format=Reply, images=build(authentic.original.body, edge),
    )

    assert reply.answer == "yes"
    assert len(calls) == 2
    assert calls[1]["messages"][0]["content"][0]["text"] == repair


@pytest.mark.asyncio
async def test_a_repair_without_room_for_its_note_leaves_the_answer_invalid(authentic):
    """A re-ask keeps no correction room; when its answer is invalid, that is the failure, not its size."""
    tokens = client().request_tokens("same prompt", response_format=Reply, images=authentic)
    calls = []
    edge = edge_client(tokens, answering(calls, '{"not_answer":true}'))

    with pytest.raises(StructuredLlmError) as raised:
        await edge.evaluate_revision_work("same prompt", response_format=Reply, images=build(authentic.original.body, edge))

    assert raised.value.terminal_category == "invalid_response"
    assert _failure_from_error(raised.value).category == "invalid_response"
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
        "fixture prompt", response_format=Reply, model=MODEL,
    )
    assert response.answer == "fake-only" and len(calls) == 1
