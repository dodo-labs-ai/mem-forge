import { useMutation } from "@tanstack/react-query";
import { FolderOpen, Loader2 } from "lucide-react";
import { errorMessage } from "@/lib/errors";
import { Button } from "@/ui/button";
import { Input } from "@/ui/input";
import { useLocalSetupJob } from "../../useLocalSetupJob";
import type { ConfigControlProps } from "../widgets";

/**
 * The folder a local markdown Source reads: a path the person types, or one
 * chosen in the native folder picker on their computer through local sync
 * (`local_markdown_pick_root`). Closing the picker without choosing keeps the
 * path as it was.
 */
export default function LocalFolderPicker({ field, id, describedBy, value, onChange, onBlur, invalid, readOnly }: ConfigControlProps) {
  const path = typeof value === "string" ? value : "";
  const pickRoot = useLocalSetupJob("local_markdown_pick_root");
  const pick = useMutation({
    mutationFn: () => pickRoot({ payload: { title: "Choose folder to sync", initial_directory: path.trim() || null } }),
    onSuccess: (result) => {
      const root = result.root?.trim();
      if (result.cancelled || !root) return;
      onChange(root);
      onBlur?.();
    },
  });

  return (
    <div className="space-y-1.5">
      <div className="flex gap-2">
        <Input
          id={id}
          aria-describedby={describedBy}
          aria-invalid={invalid || undefined}
          className="font-mono"
          autoComplete="off"
          placeholder={field.placeholder}
          value={path}
          readOnly={readOnly}
          onChange={(event) => onChange(event.target.value)}
          onBlur={onBlur}
        />
        {readOnly ? null : (
          <Button type="button" variant="outline" disabled={pick.isPending} onClick={() => pick.mutate()}>
            {pick.isPending ? <Loader2 aria-hidden className="animate-spin" /> : <FolderOpen aria-hidden />}
            {pick.isPending ? "Waiting for your choice…" : "Choose folder"}
          </Button>
        )}
      </div>
      {pick.isError ? (
        <p role="alert" className="text-xs text-tone-danger-foreground">
          {errorMessage(pick.error)}
        </p>
      ) : null}
    </div>
  );
}
