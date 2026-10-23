import { render, screen } from "@testing-library/react";
import { Suspense } from "react";
import { expect, test } from "vitest";
import { lazyRegistry } from "./lazyRegistry";

function Greeting({ name }: { name: string }) {
  return <p>Hello {name}</p>;
}

test("draws the component a name loads, and nothing for an unknown or missing name", async () => {
  const draw = lazyRegistry<{ name: string }>({ greeting: async () => ({ default: Greeting }) });

  expect(draw("unknown", { name: "Ada" })).toBeNull();
  expect(draw(null, { name: "Ada" })).toBeNull();

  render(<Suspense fallback={<p>Loading</p>}>{draw("greeting", { name: "Ada" })}</Suspense>);

  expect(await screen.findByText("Hello Ada")).toBeInTheDocument();
  expect(draw("greeting", { name: "Ada" })?.type).toBe(draw("greeting", { name: "Bo" })?.type);
});
