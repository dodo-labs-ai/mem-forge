import type { Meta, StoryObj } from "@storybook/react-vite";
import { Archive, CheckCircle2, Clock, Lock, X } from "lucide-react";
import { Button } from "@/ui/button";
import { Notice } from "./Notice";

const meta = {
  component: Notice,
  args: {
    tone: "warn",
    icon: Clock,
    title: "This proposal expired.",
    children: "The underlying memory changed before a decision was applied.",
  },
} satisfies Meta<typeof Notice>;
export default meta;
type Story = StoryObj<typeof meta>;

export const WithTitle: Story = {};

export const BodyOnly: Story = {
  args: {
    tone: "live",
    icon: Lock,
    title: undefined,
    children: "These settings come from environment variables and cannot be changed here.",
  },
};

export const WithAction: Story = {
  args: {
    tone: "idle",
    icon: Archive,
    title: "Superseded.",
    children: "A newer memory replaced this one, so searches no longer return it.",
    action: (
      <Button size="sm" variant="outline">
        Open newer memory
      </Button>
    ),
  },
};

export const Dismissible: Story = {
  args: {
    tone: "ok",
    icon: CheckCircle2,
    title: undefined,
    children: "Payroll was deleted. 12 memories moved to Unsorted.",
    action: (
      <Button size="icon-sm" variant="ghost" aria-label="Dismiss">
        <X />
      </Button>
    ),
  },
};
