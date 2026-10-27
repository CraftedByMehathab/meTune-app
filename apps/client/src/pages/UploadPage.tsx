import {
  IconAlertTriangle,
  IconArrowRight,
  IconCheck,
  IconCircleX,
  IconFileZip,
  IconFolder,
  IconLoader2,
  IconPlayerPlay,
  IconRefresh,
} from "@tabler/icons-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type DragEvent, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { PageHeading, Stat, ToneDot, Waveform } from "@/components/Brand";
import { FileStatusIcon, fileTotal, StatusBadge } from "@/components/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { api, type BatchDetail, type BatchSummary, uploadBatch } from "@/lib/api";
import { cn } from "@/lib/utils";

const STEPS = ["Upload", "Validate", "Analyze", "Review"];
const FORMATS = ["wav", "mp3", "ogg", "opus", "m4a", "flac", "webm"];

function stepOf(status: string | undefined) {
  if (!status) return 0;
  if (status === "validated" || status === "invalid") return 1;
  if (status === "running") return 2;
  return 3;
}

export function UploadPage() {
  const [params, setParams] = useSearchParams();
  const id = params.get("batch");
  const select = (next: string | null) => setParams(next ? { batch: next } : {});
  const batches = useQuery({ queryKey: ["batches"], queryFn: api.batches, refetchInterval: 5000 });
  const batch = useQuery({
    queryKey: ["batch", id],
    queryFn: () => api.batch(id!),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data?.status === "running" ? 2000 : false),
  });

  return (
    <div className="space-y-8">
      <PageHeading eyebrow="New analysis" title="Upload an evaluation batch">
        <Stepper current={stepOf(batch.data?.status)} />
      </PageHeading>
      {!id && <DropZone onUploaded={select} />}
      {id && batch.isPending && <p className="text-muted-foreground">Loading…</p>}
      {id && batch.error && <p className="text-destructive">{batch.error.message}</p>}
      {batch.data && <BatchPanel batch={batch.data} onReset={() => select(null)} />}
      {!!batches.data?.length && <RecentBatches batches={batches.data} onOpen={select} />}
    </div>
  );
}

function Stepper({ current }: { current: number }) {
  return (
    <ol className="flex items-center gap-2">
      {STEPS.map((s, i) => (
        <li key={s} className="flex items-center gap-2">
          <span
            className={cn(
              "flex size-6 items-center justify-center text-[11px] font-semibold",
              i < current ? "bg-primary text-primary-foreground" : i === current ? "bg-ink text-ink-foreground" : "bg-muted text-muted-foreground",
            )}
          >
            {i < current ? <IconCheck className="size-3.5" /> : i + 1}
          </span>
          <span className={cn("hidden text-xs font-semibold uppercase tracking-widest md:inline", i > current && "text-muted-foreground")}>
            {s}
          </span>
          {i < STEPS.length - 1 && <span className="h-px w-6 bg-border" />}
        </li>
      ))}
    </ol>
  );
}

/** Walks dropped folders so drag-and-drop matches the folder picker; paths are kept for manifest matching. */
async function filesFromDrop(e: DragEvent): Promise<File[]> {
  const out: File[] = [];
  const walk = async (entry: FileSystemEntry): Promise<void> => {
    if (entry.isFile) {
      const f = await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej));
      out.push(new File([f], entry.fullPath.replace(/^\//, ""), { type: f.type, lastModified: f.lastModified }));
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      // readEntries returns results in chunks; keep reading until it returns nothing.
      for (;;) {
        const chunk = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
        if (!chunk.length) break;
        for (const c of chunk) await walk(c);
      }
    }
  };
  const entries = Array.from(e.dataTransfer.items)
    .map((i) => i.webkitGetAsEntry())
    .filter((x): x is FileSystemEntry => !!x);
  if (!entries.length) return Array.from(e.dataTransfer.files);
  for (const entry of entries) await walk(entry);
  return out;
}

function DropZone({ onUploaded }: { onUploaded: (id: string) => void }) {
  const [over, setOver] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const zipRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);
  const qc = useQueryClient();

  async function send(files: File[]) {
    if (!files.length) return;
    setProgress(0);
    try {
      const b = await uploadBatch(files, "", setProgress);
      qc.setQueryData(["batch", b.id], b);
      qc.invalidateQueries({ queryKey: ["batches"] });
      onUploaded(b.id);
    } catch (e) {
      toast.error("Could not upload batch", { description: (e as Error).message });
    } finally {
      setProgress(null);
      if (zipRef.current) zipRef.current.value = "";
      if (folderRef.current) folderRef.current.value = "";
    }
  }

  const busy = progress !== null;
  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={async (e) => {
          e.preventDefault();
          setOver(false);
          if (!busy) send(await filesFromDrop(e));
        }}
        className={cn(
          "relative flex min-h-80 flex-col items-center justify-center gap-6 border-2 border-dashed bg-card p-10 text-center transition-colors",
          over ? "border-primary bg-primary/5" : "border-border",
        )}
      >
        {busy ? (
          <div className="flex w-full max-w-xs flex-col items-center gap-4">
            <IconLoader2 className="size-10 animate-spin text-primary" />
            <p className="font-medium">{progress < 1 ? "Uploading files…" : "Checking the manifest…"}</p>
            <Progress value={progress * 100} />
            <span className="font-mono text-xs tabular-nums text-muted-foreground">{Math.round(progress * 100)}%</span>
          </div>
        ) : (
          <>
            <div className="flex size-14 items-center justify-center bg-ink text-signal">
              <IconFolder className="size-6" />
            </div>
            <div className="space-y-1">
              <p className="font-heading text-xl font-semibold">Drop a folder or ZIP here</p>
              <p className="text-sm text-muted-foreground">
                Audio files plus an optional CSV manifest with <code className="font-mono text-foreground">name</code> and{" "}
                <code className="font-mono text-foreground">result_json</code> columns.
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-3">
              <Button onClick={() => folderRef.current?.click()}>
                <IconFolder data-icon="inline-start" /> Choose folder
              </Button>
              <Button variant="outline" onClick={() => zipRef.current?.click()}>
                <IconFileZip data-icon="inline-start" /> Choose ZIP
              </Button>
            </div>
          </>
        )}
        <input ref={zipRef} type="file" accept=".zip" hidden onChange={(e) => send(Array.from(e.target.files ?? []))} />
        <input
          ref={folderRef}
          type="file"
          hidden
          multiple
          // @ts-expect-error non-standard but supported by all major browsers
          webkitdirectory=""
          onChange={(e) => send(Array.from(e.target.files ?? []))}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Expected structure</CardTitle>
          <CardDescription>Each audio filename matches one row in the CSV, which sits at the shallowest level.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <pre className="bg-ink p-4 font-mono text-xs leading-relaxed text-ink-foreground/80">
            {"evaluation_batch/\n  call_001.wav\n  call_002.mp3\n  call_003.wav\n  "}
            <span className="text-signal">labels.csv</span>
          </pre>
          <div className="flex flex-wrap gap-x-3 gap-y-1.5">
            {FORMATS.map((f) => (
              <Badge key={f} variant="secondary">.{f}</Badge>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">Labels are optional. When present, results are scored against them.</p>
        </CardContent>
      </Card>
    </div>
  );
}

function BatchPanel({ batch: b, onReset }: { batch: BatchDetail; onReset: () => void }) {
  const qc = useQueryClient();
  const running = b.status === "running";
  const files = useQuery({
    queryKey: ["files", b.id],
    queryFn: () => api.files(b.id),
    refetchInterval: running ? 2000 : false,
  });
  const start = useMutation({
    mutationFn: () => api.start(b.id),
    onSuccess: (next) => {
      qc.setQueryData(["batch", b.id], next);
      qc.invalidateQueries({ queryKey: ["files", b.id] });
      qc.invalidateQueries({ queryKey: ["batches"] });
    },
    onError: (e) => toast.error("Could not start processing", { description: e.message }),
  });

  const c = b.counts;
  const total = fileTotal(c);
  const finished = c.done + c.error;
  const percent = total ? Math.round((finished / total) * 100) : 0;
  const audioMin = b.totals?.audio_min ?? (files.data ?? []).reduce((s, f) => s + (f.duration_s ?? 0), 0) / 60;
  const m = b.manifest;

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
      <Card className="overflow-hidden pt-0">
        <div className="flex items-center justify-between gap-4 bg-ink px-6 py-5 text-ink-foreground">
          <div className="min-w-0">
            <p className="flex items-center gap-3 text-xs uppercase tracking-widest text-ink-foreground/55">
              Batch <StatusBadge status={b.status} className="text-signal" />
            </p>
            <p className="truncate font-heading text-xl font-semibold">{b.name}</p>
          </div>
          <Waveform bars={24} animated={running} className="h-8 text-signal" />
        </div>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Files" value={total} />
            <Stat label="To process" value={m.to_process} />
            <Stat label="Labeled" value={m.has_labels ? m.matched : 0} />
            <Stat label="Audio min" value={audioMin ? audioMin.toFixed(1) : "—"} />
          </div>

          {b.status !== "validated" && b.status !== "invalid" && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="font-medium">
                  {running ? `Analyzing ${Math.min(finished + 1, total)} of ${total}…` : "Analysis complete"}
                </span>
                <span className="font-mono tabular-nums text-muted-foreground">{percent}%</span>
              </div>
              <Progress value={percent} />
            </div>
          )}

          {!!files.data?.length && (
            <ScrollArea className="h-72 border">
              <ul className="divide-y">
                {files.data.map((f) => (
                  <li key={f.id} className="flex items-center gap-3 px-4 py-2.5 text-sm">
                    <FileStatusIcon status={f.status} />
                    <span className="min-w-0 flex-1 truncate font-mono text-xs" title={f.name}>{f.name}</span>
                    {f.error && <span className="max-w-60 truncate text-xs text-destructive" title={f.error}>{f.error}</span>}
                    {f.degraded && <span className="text-xs text-warning">baseline</span>}
                    {f.result && (
                      <span className="inline-flex items-center gap-2 text-xs capitalize">
                        <ToneDot tone={f.result.emotional_tone} /> {f.result.emotional_tone}
                      </span>
                    )}
                    {(f.status === "queued" || f.status === "pending") && <span className="text-xs text-muted-foreground">Queued</span>}
                    {f.status === "running" && <span className="text-xs text-primary">Analyzing…</span>}
                  </li>
                ))}
              </ul>
            </ScrollArea>
          )}

          <div className="flex flex-wrap items-center justify-between gap-3">
            <Button variant="ghost" onClick={onReset}>
              <IconRefresh data-icon="inline-start" /> New upload
            </Button>
            {b.status === "validated" && (
              <Button size="lg" onClick={() => start.mutate()} disabled={start.isPending || !m.to_process}>
                <IconPlayerPlay data-icon="inline-start" /> Start analysis · {m.to_process} files
              </Button>
            )}
            {(b.status === "done" || b.status === "done_with_errors") && (
              <Button size="lg" asChild>
                <Link to={`/dashboard/results?batch=${b.id}`}>
                  View results <IconArrowRight data-icon="inline-end" />
                </Link>
              </Button>
            )}
            {running && <p className="text-sm text-muted-foreground">You can leave this page — processing continues.</p>}
          </div>
          {b.status === "done_with_errors" && c.error > 0 && (
            <p className="text-sm text-muted-foreground">
              {c.error} file{c.error > 1 ? "s" : ""} could not be analyzed. The rest of the batch completed normally.
            </p>
          )}
        </CardContent>
      </Card>

      <ValidationCard batch={b} />
    </div>
  );
}

function ValidationCard({ batch: b }: { batch: BatchDetail }) {
  const m = b.manifest;
  const issues = m.issues ?? [];
  const problems = issues.length + (m.fatal ? 1 : 0);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-3">
          Validation
          {problems ? <Badge variant="destructive">{problems} issues</Badge> : <Badge className="text-success">All good</Badge>}
        </CardTitle>
        <CardDescription>
          Manifest: <span className="font-mono text-foreground">{m.csv_name ?? "none"}</span>
          {m.has_labels && " · labels found, auto-scoring on"}. Issues never block the batch — affected files are skipped or
          scored as unlabeled.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          <Count label="Matched" n={m.matched} />
          <Count label="Audio without row" n={m.audio_without_row} warn />
          <Count label="Row without audio" n={m.row_without_audio} warn />
          <Count label="Unsupported" n={m.unsupported} err />
          <Count label="Bad label JSON" n={m.bad_json} warn />
        </dl>
        {m.fatal && <Issue level="error" name="Batch" message={m.fatal} />}
        {issues.length === 0 && !m.fatal ? (
          <div className="flex items-center gap-3 bg-success/10 p-4 text-sm">
            <IconCheck className="size-5 text-success" /> Every audio file matches a manifest row.
          </div>
        ) : (
          <ScrollArea className="max-h-96">
            <ul className="space-y-2">
              {issues.map((i, k) => (
                <li key={k}>
                  <Issue level={i.level} name={i.name} message={i.message} />
                </li>
              ))}
            </ul>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}

function Count({ label, n, warn, err }: { label: string; n: number; warn?: boolean; err?: boolean }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className={cn("font-mono tabular-nums", n > 0 && (err ? "text-destructive" : warn ? "text-warning" : "text-success"))}>{n}</dd>
    </div>
  );
}

function Issue({ level, name, message }: { level: string; name: string; message: string }) {
  const error = level === "error";
  return (
    <div className={cn("flex gap-3 border-l-2 p-3 text-sm", error ? "border-destructive bg-destructive/5" : "border-warning bg-warning/10")}>
      {error ? (
        <IconCircleX className="mt-0.5 size-4 shrink-0 text-destructive" />
      ) : (
        <IconAlertTriangle className="mt-0.5 size-4 shrink-0 text-warning" />
      )}
      <div className="min-w-0">
        <p className="flex flex-wrap items-center gap-2">
          <span className="truncate font-mono text-xs font-medium">{name}</span>
          <span className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">{level}</span>
        </p>
        <p className="text-muted-foreground">{message}</p>
      </div>
    </div>
  );
}

function RecentBatches({ batches, onOpen }: { batches: BatchSummary[]; onOpen: (id: string) => void }) {
  return (
    <section className="space-y-3">
      <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Recent batches</h2>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {batches.map((b) => (
          <button
            key={b.id}
            onClick={() => onOpen(b.id)}
            className="group flex items-center gap-4 border bg-card p-4 text-left transition-colors hover:border-primary"
          >
            <div className="flex size-10 shrink-0 items-center justify-center bg-muted text-muted-foreground group-hover:bg-primary group-hover:text-primary-foreground">
              <IconFolder className="size-5" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium">{b.name}</p>
              <p className="text-xs text-muted-foreground">
                {b.counts.done}/{fileTotal(b.counts)} analyzed ·{" "}
                {new Date(b.created_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}
              </p>
            </div>
            <StatusBadge status={b.status} />
          </button>
        ))}
      </div>
    </section>
  );
}
