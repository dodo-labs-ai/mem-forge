"""Resolve current Confluence body references to exact attachment entities."""

from __future__ import annotations

import csv
from html.parser import HTMLParser

from memforge.source_artifacts import (
    SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES,
    SourceArtifactContractError,
    normalize_source_artifact_media_type,
)


class _StorageArtifactReferenceParser(HTMLParser):
    def __init__(
        self,
        *,
        page_id: str,
        page_title: str,
        space_key: str,
    ) -> None:
        super().__init__(convert_charrefs=True)
        self._page_id = page_id
        self._page_title = page_title
        self._space_key = space_key
        self._image_depth = 0
        self._attachment_filename: str | None = None
        self._attachment_container_is_current: bool | None = None
        self._gallery_parameters: dict[str, str] | None = None
        self._gallery_parameter_name: str | None = None
        self._gallery_parameter_text: list[str] = []
        self.artifact_filenames: list[str] = []
        self.galleries: list[dict[str, str]] = []
        self.errors: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        values = dict(attrs)
        if tag == "ac:image":
            self._image_depth += 1
            return
        if (
            tag in {"ac:structured-macro", "ac:macro"}
            and values.get("ac:name") == "gallery"
        ):
            if self._gallery_parameters is not None:
                self.errors.append("nested_gallery")
                return
            self._gallery_parameters = {}
            return
        if tag == "ac:parameter" and self._gallery_parameters is not None:
            parameter_name = values.get("ac:name")
            if not parameter_name or self._gallery_parameter_name is not None:
                self.errors.append("invalid_gallery_parameter")
                return
            self._gallery_parameter_name = parameter_name
            self._gallery_parameter_text = []
            return
        if tag == "ri:attachment":
            filename = values.get("ri:filename")
            if not filename or self._attachment_filename is not None:
                self.errors.append("invalid_artifact_reference")
                return
            self._attachment_filename = filename
            self._attachment_container_is_current = None
            return
        if (
            self._attachment_filename is not None
            and tag in {"ri:page", "ri:blog-post", "ri:content-entity"}
        ):
            self._attachment_container_is_current = (
                (
                    tag == "ri:content-entity"
                    and values.get("ri:content-id") == self._page_id
                )
                or (
                    tag == "ri:page"
                    and values.get("ri:content-title") == self._page_title
                    and values.get("ri:space-key") in (None, "", self._space_key)
                )
            )

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        values = dict(attrs)
        if tag == "ri:attachment":
            filename = values.get("ri:filename")
            if not filename or self._attachment_filename is not None:
                self.errors.append("invalid_artifact_reference")
                return
            self.artifact_filenames.append(filename)
            return
        if (
            tag in {"ac:structured-macro", "ac:macro"}
            and values.get("ac:name") == "gallery"
        ):
            self.handle_starttag(tag, attrs)
            self.handle_endtag(tag)
            return
        if tag == "ac:parameter" and self._gallery_parameters is not None:
            self.handle_starttag(tag, attrs)
            self.handle_endtag(tag)
            return
        self.handle_starttag(tag, attrs)
        if tag == "ac:image":
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "ac:parameter" and self._gallery_parameter_name is not None:
            assert self._gallery_parameters is not None
            self._gallery_parameters[self._gallery_parameter_name] = "".join(
                self._gallery_parameter_text
            ).strip()
            self._gallery_parameter_name = None
            self._gallery_parameter_text = []
            return
        if (
            tag in {"ac:structured-macro", "ac:macro"}
            and self._gallery_parameters is not None
        ):
            if self._gallery_parameter_name is not None:
                self.errors.append("unclosed_gallery_parameter")
            else:
                self.galleries.append(dict(self._gallery_parameters))
            self._gallery_parameters = None
            self._gallery_parameter_name = None
            self._gallery_parameter_text = []
            return
        if tag == "ri:attachment" and self._attachment_filename is not None:
            if self._attachment_container_is_current is not False:
                self.artifact_filenames.append(self._attachment_filename)
            self._attachment_filename = None
            self._attachment_container_is_current = None
            return
        if tag == "ac:image" and self._image_depth:
            if self._attachment_filename is not None:
                self.errors.append("unclosed_artifact_reference")
                self._attachment_filename = None
                self._attachment_container_is_current = None
            self._image_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._gallery_parameter_name is not None:
            self._gallery_parameter_text.append(data)

    def result(self) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
        if (
            self.errors
            or self._image_depth
            or self._attachment_filename is not None
            or self._gallery_parameters is not None
            or self._gallery_parameter_name is not None
        ):
            raise SourceArtifactContractError(
                "Confluence current body Artifact references are malformed"
            )
        return (
            tuple(dict.fromkeys(self.artifact_filenames)),
            tuple(self.galleries),
        )


def _gallery_filenames(
    *,
    parameters: dict[str, str],
    image_filenames: tuple[str, ...],
) -> tuple[str, ...]:
    if any(
        parameters.get(key, "").strip()
        for key in ("page", "includeLabel", "excludeLabel")
    ):
        raise SourceArtifactContractError(
            "Confluence Gallery membership requires unsupported provider metadata"
        )

    selected = list(image_filenames)
    include = parameters.get("include", "").strip()
    if include:
        included = set(next(csv.reader([include], skipinitialspace=True)))
        selected = [filename for filename in selected if filename in included]
    exclude = parameters.get("exclude", "").strip()
    if exclude:
        excluded = set(next(csv.reader([exclude], skipinitialspace=True)))
        selected = [filename for filename in selected if filename not in excluded]
    return tuple(selected)


def resolve_current_confluence_artifacts(
    *,
    body_html: str,
    page_id: str,
    page_title: str,
    space_key: str,
    attachment_descriptors: tuple[dict, ...],
) -> tuple[dict, ...]:
    """Return exact current attachment entities reached by the current body."""

    parser = _StorageArtifactReferenceParser(
        page_id=page_id,
        page_title=page_title,
        space_key=space_key,
    )
    parser.feed(body_html)
    parser.close()
    referenced_filenames, local_galleries = parser.result()

    descriptors_by_title: dict[str, list[dict]] = {}
    for descriptor in attachment_descriptors:
        descriptors_by_title.setdefault(
            str(descriptor.get("title") or ""),
            [],
        ).append(descriptor)

    gallery_image_filenames = tuple(
        str(descriptor.get("title") or "")
        for descriptor in attachment_descriptors
        if normalize_source_artifact_media_type(
            (
                descriptor.get("extensions")
                if isinstance(descriptor.get("extensions"), dict)
                else {}
            ).get("mediaType")
        ).startswith("image/")
    )
    referenced_filenames = tuple(
        dict.fromkeys(
            (
                *referenced_filenames,
                *(
                    filename
                    for parameters in local_galleries
                    for filename in _gallery_filenames(
                        parameters=parameters,
                        image_filenames=gallery_image_filenames,
                    )
                ),
            )
        )
    )

    selected: list[dict] = []
    for filename in referenced_filenames:
        matches = descriptors_by_title.get(filename, [])
        if len(matches) != 1:
            raise SourceArtifactContractError(
                "Confluence current body Artifact reference cannot be resolved uniquely"
            )
        descriptor = matches[0]
        extensions = (
            descriptor.get("extensions")
            if isinstance(descriptor.get("extensions"), dict)
            else {}
        )
        if (
            normalize_source_artifact_media_type(extensions.get("mediaType"))
            in SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES
        ):
            selected.append(descriptor)
    return tuple(selected)
