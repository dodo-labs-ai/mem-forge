import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { SegmentedControl, type SegmentedOption } from "./SegmentedControl";

type Kind = "all" | "lifecycle" | "memory";

const KINDS: SegmentedOption<Kind>[] = [
  { value: "all", label: "All decisions" },
  { value: "lifecycle", label: "Source lifecycle" },
  { value: "memory", label: "Memory updates" },
];

const WINDOWS: SegmentedOption<string>[] = [
  { value: "1", label: "24h", accessibleLabel: "Last 24 hours" },
  { value: "7", label: "7d", accessibleLabel: "Last 7 days" },
  { value: "30", label: "30d", accessibleLabel: "Last 30 days" },
];

const meta = {
  component: SegmentedControl<string>,
  args: { options: KINDS, value: "all", onValueChange: () => {}, "aria-label": "Show" },
} satisfies Meta<typeof SegmentedControl<string>>;
export default meta;
type Story = StoryObj<typeof meta>;

function Controlled({ options, initial }: { options: SegmentedOption<string>[]; initial: string }) {
  const [value, setValue] = useState(initial);
  return <SegmentedControl aria-label="Show" options={options} value={value} onValueChange={setValue} />;
}

export const Labels: Story = {
  args: { "aria-label": "Show" },
  render: () => <Controlled options={KINDS} initial="all" />,
};

export const Abbreviated: Story = {
  args: { "aria-label": "Window" },
  render: () => <Controlled options={WINDOWS} initial="1" />,
};
