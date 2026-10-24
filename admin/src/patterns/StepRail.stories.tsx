import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { StepRail, type RailStep } from "./StepRail";

const STEPS: RailStep[] = [
  { key: "basics", label: "Basics", complete: true, summary: "Payroll wiki" },
  { key: "connection", label: "Confluence site", complete: true, summary: "wiki.example.com" },
  { key: "scope", label: "Pages to sync", errorCount: 1 },
  { key: "project", label: "Project" },
  { key: "schedule", label: "Schedule", complete: true, summary: "Every 60 minutes" },
];

const meta = {
  component: StepRail,
  args: { steps: STEPS, current: "scope", onSelect: () => {}, "aria-label": "Setup steps" },
} satisfies Meta<typeof StepRail>;
export default meta;
type Story = StoryObj<typeof meta>;

function Controlled() {
  const [current, setCurrent] = useState("scope");
  return (
    <div className="w-64">
      <StepRail aria-label="Setup steps" steps={STEPS} current={current} onSelect={setCurrent} />
    </div>
  );
}

export const Steps: Story = { render: () => <Controlled /> };
