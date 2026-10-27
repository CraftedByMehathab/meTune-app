import { IconChevronDown, IconUpload } from "@tabler/icons-react";
import { useQuery } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { StatusBadge } from "@/components/StatusBadge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { api, type BatchSummary } from "@/lib/api";

/**
 * The batch shown on Results / Reports: `?batch=` if set, else the newest batch with any analyzed files.
 * Returns `undefined` while the list is loading and `null` when there is nothing to show.
 */
export function useSelectedBatch() {
  const [params, setParams] = useSearchParams();
  const batches = useQuery({ queryKey: ["batches"], queryFn: api.batches });
  const explicit = params.get("batch");
  const list = batches.data;
  const id = explicit ?? (list ? (list.find((b) => b.counts.done > 0)?.id ?? null) : undefined);
  return { id, batches: list ?? [], select: (next: string) => setParams({ batch: next }) };
}

export function BatchPicker({ current, batches, onSelect }: { current?: string; batches: BatchSummary[]; onSelect: (id: string) => void }) {
  const active = batches.find((b) => b.id === current);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline">
          {active?.name ?? "Choose batch"} <IconChevronDown data-icon="inline-end" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-72">
        <DropdownMenuLabel>Batches</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {batches.map((b) => (
          <DropdownMenuItem key={b.id} onClick={() => onSelect(b.id)} className="justify-between gap-3">
            <span className="truncate">{b.name}</span>
            <StatusBadge status={b.status} />
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function NoBatch() {
  return (
    <div className="flex min-h-64 flex-col items-center justify-center gap-4 border-2 border-dashed bg-card p-10 text-center">
      <p className="font-heading text-xl font-semibold">No analyzed batches yet</p>
      <p className="text-sm text-muted-foreground">Upload a batch and run the analysis to see results here.</p>
      <Button asChild>
        <Link to="/dashboard">
          <IconUpload data-icon="inline-start" /> Upload a batch
        </Link>
      </Button>
    </div>
  );
}
