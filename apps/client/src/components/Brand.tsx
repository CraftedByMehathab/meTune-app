import type { Result } from "@/lib/api";
import { cn } from "@/lib/utils";

/** Decorative bar waveform; heights are deterministic so SSR/hydration and re-renders match. */
export function Waveform({ bars = 28, className, animated = true }: { bars?: number; className?: string; animated?: boolean }) {
  return (
    <div className={cn("flex h-10 items-center gap-[3px]", className)} aria-hidden>
      {Array.from({ length: bars }, (_, i) => {
        const h = 25 + Math.abs(Math.sin(i * 1.7) * 60) + (i % 4) * 4;
        return (
          <span
            key={i}
            className={cn("w-[3px] rounded-full bg-current", animated && "animate-wave")}
            style={{ height: `${Math.min(h, 100)}%`, animationDelay: `${(i % 9) * 0.11}s` }}
          />
        );
      })}
    </div>
  );
}

export function Logo({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <div className="flex size-8 items-center justify-center bg-signal text-signal-foreground">
        <Waveform bars={5} className="h-4 gap-[2px]" />
      </div>
      <span className="font-heading text-lg font-semibold tracking-tight">
        me<span className="text-signal">Tune</span>
      </span>
    </div>
  );
}

type Tone = Result["emotional_tone"];

const TONE_BG: Record<Tone, string> = {
  neutral: "bg-tone-neutral",
  satisfied: "bg-tone-satisfied",
  frustrated: "bg-tone-frustrated",
  upset: "bg-tone-upset",
  distressed: "bg-tone-distressed",
};

export const ToneDot = ({ tone, className }: { tone: Tone; className?: string }) => (
  <span className={cn("inline-block size-2 shrink-0 rounded-full", TONE_BG[tone], className)} />
);

export const ToneLabel = ({ tone }: { tone: Tone }) => (
  <span className="inline-flex items-center gap-2 text-sm capitalize">
    <ToneDot tone={tone} /> {tone}
  </span>
);

/** Small uppercase eyebrow + heading used at the top of each page. */
export function PageHeading({ eyebrow, title, children }: { eyebrow: string; title: string; children?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="space-y-1">
        <p className="text-xs font-semibold uppercase tracking-widest text-primary">{eyebrow}</p>
        <h1 className="font-heading text-3xl font-semibold tracking-tight">{title}</h1>
      </div>
      {children}
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div>
      <div className="font-heading text-2xl font-semibold tabular-nums">{value}</div>
      <div className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">{label}</div>
      {hint && <div className="text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}
