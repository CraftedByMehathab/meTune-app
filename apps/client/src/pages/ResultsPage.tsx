import { IconDownload } from "@tabler/icons-react";
import { useQuery } from "@tanstack/react-query";
import { BatchPicker, NoBatch, useSelectedBatch } from "@/components/BatchPicker";
import { PageHeading, Stat } from "@/components/Brand";
import { FilesTable } from "@/components/FilesTable";
import { fileTotal } from "@/components/StatusBadge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";

export function ResultsPage() {
  const { id, batches, select } = useSelectedBatch();
  const batch = useQuery({
    queryKey: ["batch", id],
    queryFn: () => api.batch(id!),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data?.status === "running" ? 2000 : false),
  });
  const running = batch.data?.status === "running";
  const files = useQuery({
    queryKey: ["files", id],
    queryFn: () => api.files(id!),
    enabled: !!id,
    refetchInterval: running ? 2000 : false,
  });

  if (id === undefined) return <p className="text-muted-foreground">Loading…</p>;

  const b = batch.data;
  const done = (files.data ?? []).filter((f) => f.result);
  const avgConf = done.length ? done.reduce((s, f) => s + f.result!.confidence, 0) / done.length : null;
  const mismatches = done.filter((f) => f.label && (Object.keys(f.label) as (keyof typeof f.label)[]).some((k) => k !== "confidence" && f.label![k] !== f.result![k])).length;
  const labeled = done.filter((f) => f.label).length;

  return (
    <div className="space-y-8">
      <PageHeading eyebrow="Results" title={b?.name ?? "Per-call results"}>
        <div className="flex flex-wrap gap-2">
          {!!batches.length && <BatchPicker current={id ?? undefined} batches={batches} onSelect={select} />}
          {id && !!b?.counts.done && (
            <>
              <Button variant="outline" asChild>
                <a href={api.downloadUrl(id, "csv")}><IconDownload data-icon="inline-start" /> CSV</a>
              </Button>
              <Button variant="outline" asChild>
                <a href={api.downloadUrl(id, "json")}><IconDownload data-icon="inline-start" /> JSON</a>
              </Button>
            </>
          )}
        </div>
      </PageHeading>

      {id === null && <NoBatch />}
      {batch.error && <p className="text-destructive">{batch.error.message}</p>}

      {b && (
        <>
          <Card size="sm">
            <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-5">
              <Stat label="Analyzed" value={`${b.counts.done}/${fileTotal(b.counts)}`} />
              <Stat label="Failed" value={b.counts.error} />
              <Stat label="Degraded" value={(files.data ?? []).filter((f) => f.degraded).length} hint="DSP baseline only" />
              <Stat label="Avg confidence" value={avgConf != null ? avgConf.toFixed(2) : "—"} hint="vote agreement" />
              <Stat label="Label mismatches" value={labeled ? `${mismatches}/${labeled}` : "—"} hint="files with ≥1 wrong field" />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Calls</CardTitle>
              <CardDescription>
                {b.has_labels ? "Cells in red disagree with the label; hover to see the expected value." : "No labels in this batch."}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {files.data?.length ? (
                <FilesTable files={files.data} />
              ) : (
                <p className="text-sm text-muted-foreground">{files.isPending ? "Loading…" : "No files in this batch."}</p>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
