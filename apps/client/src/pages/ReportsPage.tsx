import { useQuery } from "@tanstack/react-query";
import { BatchPicker, NoBatch, useSelectedBatch } from "@/components/BatchPicker";
import { PageHeading, Stat } from "@/components/Brand";
import { Scorecard } from "@/components/Scorecard";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import { cn, usd } from "@/lib/utils";

const COST_CEILING = 0.003;

export function ReportsPage() {
  const { id, batches, select } = useSelectedBatch();
  const batch = useQuery({
    queryKey: ["batch", id],
    queryFn: () => api.batch(id!),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data?.status === "running" ? 2000 : false),
  });

  if (id === undefined) return <p className="text-muted-foreground">Loading…</p>;
  const b = batch.data;
  const t = b?.totals;
  const overBudget = t?.cost_per_min != null && t.cost_per_min > COST_CEILING;

  return (
    <div className="space-y-8">
      <PageHeading eyebrow="Reports" title={b?.name ?? "Batch report"}>
        {!!batches.length && <BatchPicker current={id ?? undefined} batches={batches} onSelect={select} />}
      </PageHeading>

      {id === null && <NoBatch />}
      {batch.error && <p className="text-destructive">{batch.error.message}</p>}

      {b && (
        <>
          <Card className="overflow-hidden pt-0">
            <div className="bg-ink px-6 py-4 font-mono text-xs text-ink-foreground/70">
              {b.model_id ? (
                <>
                  model <span className="text-signal">{b.model_id}</span> · config {b.config_hash} · build {b.git_sha}
                </>
              ) : (
                "Not started"
              )}
            </div>
            <CardContent className="grid grid-cols-2 gap-6 sm:grid-cols-3 lg:grid-cols-6">
              <Stat label="Audio" value={t ? `${t.audio_min} min` : "—"} />
              <Stat label="Total cost" value={usd(t?.cost_usd, 4)} />
              <Stat
                label="Cost / audio-min"
                value={<span className={cn(overBudget ? "text-destructive" : t?.cost_per_min != null && "text-success")}>{usd(t?.cost_per_min)}</span>}
                hint={`ceiling $${COST_CEILING}`}
              />
              <Stat label="Latency / audio-min" value={t?.latency_s_per_min != null ? `${t.latency_s_per_min}s` : "—"} />
              <Stat label="Latency p50" value={t?.p50_latency_s != null ? `${t.p50_latency_s}s` : "—"} hint="per file" />
              <Stat label="Latency p95" value={t?.p95_latency_s != null ? `${t.p95_latency_s}s` : "—"} hint="per file" />
            </CardContent>
          </Card>

          {b.metrics ? (
            <Scorecard metrics={b.metrics} />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Scorecard</CardTitle>
                <CardDescription>
                  {b.has_labels
                    ? "Scores appear once the batch finishes processing."
                    : "This batch has no labels. Add a result_json column to the manifest to score it automatically."}
                </CardDescription>
              </CardHeader>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
