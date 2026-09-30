import type { Blocker } from "react-router-dom";
import { Button } from "@/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/ui/dialog";
import { endpointList } from "./model/settingsForm";
import type { EndpointKind } from "./model/types";

interface LeaveDialogProps {
  blocker: Blocker;
  changed: EndpointKind[];
}

/** Asks before an in-app link leaves the page with unsaved changes. */
export function LeaveDialog({ blocker, changed }: LeaveDialogProps) {
  function stay() {
    if (blocker.state === "blocked") blocker.reset();
  }
  function leave() {
    if (blocker.state === "blocked") blocker.proceed();
  }

  return (
    <Dialog open={blocker.state === "blocked"} onOpenChange={(open) => (open ? undefined : stay())}>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>Leave without saving?</DialogTitle>
          <DialogDescription>
            Your changes in {endpointList(changed)} are not saved. They are lost if you leave this page.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={stay}>
            Keep editing
          </Button>
          <Button variant="destructive" onClick={leave}>
            Leave without saving
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
