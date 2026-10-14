import { AlertTriangle } from "lucide-react";
import { Notice } from "@/patterns";
import { Button } from "@/ui/button";
import { AUTH_LABELS } from "./model/providerFields";
import type { ModelProvider } from "./model/types";

interface EndpointConflictNoticeProps {
  /** The base URL the model sends its requests to. */
  endpoint: string;
  provider: ModelProvider;
  /** The providers that reach a server without a key, which keep the base URL of a server that needs none. */
  keylessProviders: readonly ModelProvider[];
  /** Gives up the base URL for the provider's own address. */
  onUseOwnAddress: () => void;
}

/**
 * Says that the model pairs a base URL with credentials from the server
 * environment, which Settings does not save, and offers the ways to keep one
 * of the two. The base URL is given up only when the user asks for it here.
 */
export function EndpointConflictNotice({ endpoint, provider, keylessProviders, onUseOwnAddress }: EndpointConflictNoticeProps) {
  const ownAddress = provider.default_endpoint;
  const keepsKey = provider.auth_strategies.includes("api_key");
  const keyless = keylessProviders.map((choice) => choice.label).join(" or ");
  return (
    <Notice
      tone="warn"
      icon={AlertTriangle}
      title="This base URL cannot be saved with credentials from the server environment"
      action={
        <Button type="button" variant="outline" size="sm" onClick={onUseOwnAddress}>
          {ownAddress === null ? "Remove the base URL" : "Use the provider's address"}
        </Button>
      }
    >
      <p>
        This model sends its requests to <span className="font-mono text-[13px] break-all">{endpoint}</span>. Settings saves
        credentials from the server environment only for{" "}
        {ownAddress === null ? "a route without a base URL" : <span className="font-mono text-[13px] break-all">{ownAddress}</span>},
        so that a base URL entered here never sends them to another server.
      </p>
      <p>
        {keepsKey ? `To keep the base URL, choose ${AUTH_LABELS.api_key} under Authentication and enter the key of that server. ` : ""}
        {keyless ? `For a server that needs no key, choose ${keyless} as the provider. ` : ""}
        {ownAddress === null
          ? "To keep the credentials, remove the base URL, and the route decides where requests go."
          : "To keep the credentials, use the provider's address."}
      </p>
    </Notice>
  );
}
