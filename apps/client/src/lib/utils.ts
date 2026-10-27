export { cn } from "cn";

export const pct = (v: number) => `${Math.round(v * 100)}%`;
export const usd = (v: number | null | undefined, digits = 5) => (v == null ? "—" : `$${v.toFixed(digits)}`);
