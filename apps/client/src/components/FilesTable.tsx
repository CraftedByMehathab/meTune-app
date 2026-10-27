import { ToneLabel } from "@/components/Brand";
import { FileStatusIcon } from "@/components/StatusBadge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { FileOut, Result } from "@/lib/api";
import { cn } from "@/lib/utils";

const yn = (v: boolean) => (v ? "yes" : "no");

/** Free text, so not scored (matches SCORED_FIELDS in the pipeline); the label is still shown on hover. */
const UNSCORED: (keyof Result)[] = ["background_noise_type", "confidence"];

/** Highlights a predicted value that disagrees with the label, when a label exists. */
function Cell({ f, k, render }: { f: FileOut; k: keyof Result; render?: (v: never) => React.ReactNode }) {
  if (!f.result) return <TableCell className="text-muted-foreground">—</TableCell>;
  const v = f.result[k];
  const shown = render ? render(v as never) : String(v);
  if (UNSCORED.includes(k)) {
    const hint = f.label && f.label[k] !== v ? `label: ${String(f.label[k]) || "—"}` : undefined;
    return <TableCell className="capitalize" title={hint}>{shown || "—"}</TableCell>;
  }
  const miss = f.label != null && f.label[k] !== v;
  return (
    <TableCell
      className={cn("capitalize", miss && "bg-destructive/5 text-destructive")}
      title={miss ? `label: ${String(f.label![k])}` : undefined}
    >
      {shown || "—"}
    </TableCell>
  );
}

export function FilesTable({ files }: { files: FileOut[] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>File</TableHead>
          <TableHead>Tone</TableHead>
          <TableHead>Intensity</TableHead>
          <TableHead>Noise</TableHead>
          <TableHead>Type</TableHead>
          <TableHead>Severity</TableHead>
          <TableHead>Quality</TableHead>
          <TableHead>Overlap</TableHead>
          <TableHead>Long silence</TableHead>
          <TableHead className="text-right">Conf.</TableHead>
          <TableHead className="text-right">Latency</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {files.map((f) => (
          <TableRow key={f.id}>
            <TableCell>
              <div className="flex items-center gap-2">
                <FileStatusIcon status={f.status} />
                <div className="min-w-0">
                  <div className="max-w-56 truncate font-mono text-xs" title={f.name}>{f.name}</div>
                  {f.degraded && <div className="text-xs text-warning">LLM failed, baseline</div>}
                  {f.error && <div className="max-w-56 truncate text-xs text-destructive" title={f.error}>{f.error}</div>}
                </div>
              </div>
            </TableCell>
            <Cell f={f} k="emotional_tone" render={(v: Result["emotional_tone"]) => <ToneLabel tone={v} />} />
            <Cell f={f} k="emotional_intensity" />
            <Cell f={f} k="background_noise_present" render={yn} />
            <Cell f={f} k="background_noise_type" />
            <Cell f={f} k="background_noise_severity" />
            <Cell f={f} k="audio_quality" render={(v: string) => v.replace("_", " ")} />
            <Cell f={f} k="speaker_overlap_present" render={yn} />
            <Cell f={f} k="long_silence_present" render={yn} />
            <TableCell className="text-right font-mono tabular-nums">{f.result?.confidence.toFixed(2) ?? "—"}</TableCell>
            <TableCell className="text-right font-mono tabular-nums">
              {f.latency_ms != null ? `${(f.latency_ms / 1000).toFixed(1)}s` : "—"}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
