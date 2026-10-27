import { IconCheck, IconCircleX, IconLoader2 } from "@tabler/icons-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

/** Batch status: validated | invalid | running | done | done_with_errors. */
const BATCH_CLASS: Record<string, string> = {
  validated: "text-primary",
  running: "text-primary",
  done: "text-success",
  done_with_errors: "text-warning",
  invalid: "text-destructive",
};

export const StatusBadge = ({ status, className }: { status: string; className?: string }) => (
  <Badge variant="secondary" className={cn(BATCH_CLASS[status], className)}>
    {status.replace(/_/g, " ")}
  </Badge>
);

/** File status icon: pending | queued | running | done | error. */
export function FileStatusIcon({ status }: { status: string }) {
  if (status === "done") return <IconCheck className="size-4 shrink-0 text-success" />;
  if (status === "error") return <IconCircleX className="size-4 shrink-0 text-destructive" />;
  if (status === "running") return <IconLoader2 className="size-4 shrink-0 animate-spin text-primary" />;
  return <span className="mx-1 size-2 shrink-0 rounded-full bg-border" />;
}

export const fileTotal = (c: { pending: number; queued: number; running: number; done: number; error: number }) =>
  c.pending + c.queued + c.running + c.done + c.error;
