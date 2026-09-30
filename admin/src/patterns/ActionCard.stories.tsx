import type { Meta, StoryObj } from "@storybook/react-vite";
import { Button } from "@/ui/button";
import { ActionCard } from "./ActionCard";

const meta = {
  component: ActionCard,
  args: {
    tone: "danger",
    title: "Payroll Jira",
    description: "Sign-in required",
    action: (
      <Button size="sm" variant="outline">
        Sign in
      </Button>
    ),
  },
} satisfies Meta<typeof ActionCard>;
export default meta;
type Story = StoryObj<typeof meta>;

export const NeedsAction: Story = {};
export const Warning: Story = { args: { tone: "warn", description: "Scheduled sync is overdue" } };
