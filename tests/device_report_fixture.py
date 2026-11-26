"""Device reports for tests that lease local-agent jobs."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from memforge.local_agent.device_report import DeviceReport, declared_acquisition_contracts


def device_report(
    source_ids: Collection[str] = (),
    *,
    device_id: str = "dev-laptop",
    name: str = "Laptop",
    version: str = "0.2.0",
    acquisition_contracts: Mapping[str, int] | None = None,
) -> DeviceReport:
    """A device that acquires under the server's own contracts unless the test names others."""

    return DeviceReport(
        device_id=device_id,
        name=name,
        version=version,
        acquisition_contracts=(
            declared_acquisition_contracts() if acquisition_contracts is None else acquisition_contracts
        ),
        source_ids=frozenset(source_ids),
    )


def leasing_device(source_ids: Collection[str] = ()) -> dict[str, object]:
    """The device and server contracts a store lease takes: a device of the server's contracts naming ``source_ids``."""

    return {"device": device_report(source_ids), "acquisition_contracts": declared_acquisition_contracts()}


DAEMON_DEVICE = device_report().to_payload()
"""The report a daemon that names no device Source sends with its leases and heartbeats."""
