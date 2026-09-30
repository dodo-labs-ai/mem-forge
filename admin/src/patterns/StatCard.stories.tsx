import type { Meta, StoryObj } from "@storybook/react-vite";
import { StatCard } from "./StatCard";

const meta = {
  component: StatCard,
  args: { label: "Active memories", value: "3,344", detail: "412 superseded or retired" },
} satisfies Meta<typeof StatCard>;
export default meta;
type Story = StoryObj<typeof meta>;

export const Plain: Story = {};

export const NeedsAttention: Story = {
  args: { label: "Affected sources", value: "3 / 10", detail: "Sources with failures or coverage gaps", tone: "danger" },
};

export const Linked: Story = {
  args: {
    label: "Waiting for review",
    value: "5",
    tone: "warn",
    detail: "Current memories stay searchable until you decide",
    link: { label: "Review queue", render: <a href="#review" /> },
  },
};

export const Loading: Story = { args: { value: undefined } };
