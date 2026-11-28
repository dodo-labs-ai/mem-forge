"""Device reports for tests that lease local-agent jobs."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from memforge.local_agent.device_report import (
    READY,
    DeviceReport,
    DeviceSourceReadiness,
    declared_acquisition_contracts,
)

DEVICE_ID = "dev-laptop"


def device_report(
    ready: Collection[str] = (),
    *,
    not_ready: Mapping[str, DeviceSourceReadiness] | None = None,
    device_id: str = DEVICE_ID,
    name: str = "Laptop",
    version: str = "0.2.0",
    acquisition_contracts: Mapping[str, int] | None = None,
) -> DeviceReport:
    """A device that reports ``ready`` ready and ``not_ready`` with their reasons, under the server's contracts
    unless the test names others."""

    return DeviceReport(
        device_id=device_id,
        name=name,
        version=version,
        acquisition_contracts=(
            declared_acquisition_contracts() if acquisition_contracts is None else acquisition_contracts
        ),
        sources={**{source_id: READY for source_id in ready}, **(not_ready or {})},
    )


def leasing_device(ready: Collection[str] = ()) -> dict[str, object]:
    """The device and server contracts a store lease takes: a device of the server's contracts, ready for ``ready``."""

    return {"device": device_report(ready), "acquisition_contracts": declared_acquisition_contracts()}


DAEMON_DEVICE = device_report().to_payload()
"""The report a daemon that executes no device Source sends with its leases and heartbeats."""


def connect_device(database, user_id: str, report: DeviceReport | None = None, *, seen=None) -> DeviceReport:
    """Record ``report``, the user's laptop by default, as a device of the user, as its daemon's lease at ``seen`` does."""

    import asyncio

    report = report or device_report()
    asyncio.run(
        database.lease_local_agent_jobs(
            user_id=user_id,
            device=report,
            acquisition_contracts=declared_acquisition_contracts(),
            limit=0,
            lease_seconds=60,
            now=seen,
        )
    )
    return report
