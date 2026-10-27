import { IconArrowRight, IconLock, IconShieldCheck } from "@tabler/icons-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Logo, Waveform } from "@/components/Brand";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";

const FACTS = [
  ["9", "output fields"],
  ["<$0.003", "per minute"],
  ["0", "audio leaves infra"],
];

export function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const qc = useQueryClient();
  const navigate = useNavigate();
  const login = useMutation({
    mutationFn: () => api.login(username, password),
    onSuccess: (me) => {
      qc.setQueryData(["me"], me);
      navigate("/dashboard");
    },
  });

  const submit = (e: FormEvent) => {
    e.preventDefault();
    login.mutate();
  };

  return (
    <div className="grid min-h-screen lg:grid-cols-[1.1fr_1fr]">
      <aside className="relative hidden overflow-hidden bg-ink-gradient p-12 text-ink-foreground lg:flex lg:flex-col">
        <div className="absolute inset-0 grid-lines" />
        <Logo className="relative" />
        <div className="relative mt-auto space-y-8">
          <Waveform bars={56} className="h-28 text-signal" />
          <h1 className="font-heading text-5xl font-semibold leading-[1.05] tracking-tight">
            Heard.
            <br />
            Understood.
            <br />
            <span className="text-signal">Attuned.</span>
          </h1>
          <p className="max-w-md text-ink-foreground/70">
            Classify emotional tone, background noise and audio quality across entire call batches — in minutes, under
            $0.003 per audio minute.
          </p>
          <div className="grid max-w-md grid-cols-3 gap-6 border-t border-ink-foreground/15 pt-6 text-sm">
            {FACTS.map(([value, label]) => (
              <div key={label}>
                <div className="font-heading text-2xl font-semibold">{value}</div>
                <div className="text-xs uppercase tracking-widest text-ink-foreground/55">{label}</div>
              </div>
            ))}
          </div>
        </div>
      </aside>

      <main className="flex items-center justify-center p-6 sm:p-12">
        <div className="w-full max-w-sm space-y-8">
          <Logo className="lg:hidden" />
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-widest text-primary">Evaluator access</p>
            <h2 className="font-heading text-3xl font-semibold tracking-tight">Sign in to your workspace</h2>
            <p className="text-sm text-muted-foreground">Use the credentials provided by the meTune team.</p>
          </div>
          <form onSubmit={submit} className="space-y-5">
            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input id="username" autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            {login.error && <p className="text-sm text-destructive">{login.error.message}</p>}
            <Button type="submit" size="lg" className="w-full" disabled={login.isPending}>
              {login.isPending ? "Signing in…" : "Sign in"}
              <IconArrowRight data-icon="inline-end" />
            </Button>
          </form>
          <div className="space-y-3 border-t pt-6 text-xs text-muted-foreground">
            <p className="flex items-center gap-2">
              <IconShieldCheck className="size-4 text-success" /> Audio is processed in a private environment.
            </p>
            <p className="flex items-center gap-2">
              <IconLock className="size-4" /> Sessions expire after 12 hours.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
