import { ToneLabel } from "@/components/Brand";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { Metrics, Result } from "@/lib/api";
import { pct } from "@/lib/utils";

const label = (field: string) => field.replace(/_/g, " ");

export function Scorecard({ metrics: m }: { metrics: Metrics }) {
  const cm = m.tone_confusion;
  const max = Math.max(1, ...cm.matrix.flat());
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Field accuracy</CardTitle>
          <CardDescription>
            {m.n} labeled file(s). {m.n < 30 && "Small sample: treat as a smoke test, not validation."}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-x-10 gap-y-4 sm:grid-cols-2">
          {Object.entries(m.per_field).map(([f, s]) => (
            <div key={f} className="space-y-1.5">
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <span className="capitalize">{label(f)}</span>
                <span className="font-mono text-xs tabular-nums text-muted-foreground">
                  <span className="text-foreground">{pct(s.accuracy)}</span> · F1 {s.macro_f1.toFixed(2)}
                </span>
              </div>
              <div className="h-1.5 bg-muted">
                <div className="h-full bg-primary" style={{ width: pct(s.accuracy) }} />
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Tone confusion</CardTitle>
          <CardDescription>Rows are the true tone, columns the predicted tone.</CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="text-xs">
            <thead>
              <tr>
                <th />
                {cm.labels.map((l) => (
                  <th key={l} className="px-1.5 pb-2 font-semibold uppercase tracking-widest text-muted-foreground">
                    {l.slice(0, 4)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cm.matrix.map((row, i) => (
                <tr key={i}>
                  <td className="pr-3 text-right capitalize text-muted-foreground">{cm.labels[i]}</td>
                  {row.map((v, j) => (
                    <td
                      key={j}
                      className={`size-11 border text-center font-mono tabular-nums ${i === j ? "font-semibold" : ""}`}
                      style={{
                        background: v
                          ? `color-mix(in oklch, var(${i === j ? "--primary" : "--destructive"}) ${Math.round((v / max) * 55)}%, transparent)`
                          : undefined,
                      }}
                    >
                      {v || ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Per-tone scores</CardTitle>
          <CardDescription>Precision, recall and F1 for each emotional tone.</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Tone</TableHead>
                <TableHead className="text-right">P</TableHead>
                <TableHead className="text-right">R</TableHead>
                <TableHead className="text-right">F1</TableHead>
                <TableHead className="text-right">n</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {m.tone_report.map((r) => (
                <TableRow key={r.label}>
                  <TableCell><ToneLabel tone={r.label as Result["emotional_tone"]} /></TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{r.precision.toFixed(2)}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{r.recall.toFixed(2)}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{r.f1.toFixed(2)}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{r.support}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
