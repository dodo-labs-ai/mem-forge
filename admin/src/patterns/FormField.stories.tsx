import type { Meta, StoryObj } from "@storybook/react-vite";
import { Input } from "@/ui/input";
import { describedBy } from "./describedBy";
import { FormField } from "./FormField";

const meta = {
  component: FormField,
  args: { label: "Confluence Base URL", htmlFor: "base-url", children: null },
} satisfies Meta<typeof FormField>;
export default meta;
type Story = StoryObj<typeof meta>;

export const WithHint: Story = {
  render: () => (
    <FormField label="Confluence Base URL" htmlFor="base-url" hint="The site address, such as https://wiki.example.com" hintId="base-url-hint">
      <Input id="base-url" aria-describedby={describedBy("base-url-hint")} placeholder="https://wiki.example.com" />
    </FormField>
  ),
};

export const WithError: Story = {
  render: () => (
    <FormField
      label="Confluence Base URL"
      htmlFor="base-url-error"
      hint="The site address, such as https://wiki.example.com"
      hintId="base-url-error-hint"
      error="Confluence Base URL is required"
      errorId="base-url-error-message"
    >
      <Input
        id="base-url-error"
        aria-invalid
        aria-describedby={describedBy("base-url-error-message", "base-url-error-hint")}
      />
    </FormField>
  ),
};
