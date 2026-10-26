import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { expect, test } from "vitest";
import { makeRelatedMemory } from "@/test/memoryFixtures";
import { formatSourceTime } from "@/lib/format";
import { RelatedMemorySummary } from "./RelationCells";

function renderSummary(evidenceTime: string | null) {
  render(
    <MemoryRouter>
      <RelatedMemorySummary memory={makeRelatedMemory({ evidence_time: evidenceTime })} typeLabels={{}} />
    </MemoryRouter>,
  );
}

test("a related Memory recorded at an instant shows its date and time", () => {
  renderSummary("2026-03-03T12:30:00+00:00");

  const recorded = screen.getByText(/^Recorded /);
  expect(recorded).toHaveTextContent(`Recorded ${formatSourceTime("2026-03-03T12:30:00+00:00")}`);
  expect(recorded.textContent).not.toContain("+00:00");
  expect(recorded.textContent).toMatch(/2026, \d{1,2}:\d{2}/);
});

test("a related Memory recorded on a date shows the date alone", () => {
  renderSummary("2026-03-03");

  expect(screen.getByText(/^Recorded /)).toHaveTextContent(/^Recorded Mar 3, 2026$/);
});

test("a related Memory without a source time shows none", () => {
  renderSummary(null);

  expect(screen.queryByText(/^Recorded /)).not.toBeInTheDocument();
});
