import { IconChartBar, IconLogout, IconTable, IconUpload } from "@tabler/icons-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, Navigate, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { Logo } from "@/components/Brand";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { api, ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/dashboard", label: "Upload", icon: IconUpload, end: true },
  { to: "/dashboard/results", label: "Results", icon: IconTable, end: false },
  { to: "/dashboard/reports", label: "Reports", icon: IconChartBar, end: false },
];

/** Auth gate for the UI only. The API enforces access on every route. */
export function Layout() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const { search } = useLocation();
  const me = useQuery({ queryKey: ["me"], queryFn: api.me });
  const logout = useMutation({
    mutationFn: api.logout,
    onSuccess: () => {
      qc.clear();
      navigate("/login");
    },
  });

  if (me.error instanceof ApiError && me.error.status === 401) return <Navigate to="/login" replace />;
  if (me.isPending) return null;

  // Carry ?batch= across tabs so Upload → Results → Reports stay on the same batch.
  const batch = new URLSearchParams(search).get("batch");
  const qs = batch ? `?batch=${encodeURIComponent(batch)}` : "";
  const username = me.data?.username ?? "";

  return (
    <div className="min-h-screen bg-muted/40">
      <header className="sticky top-0 z-30 border-b bg-ink text-ink-foreground">
        <div className="mx-auto flex h-16 max-w-7xl items-center gap-8 px-4 sm:px-6">
          <Link to="/dashboard">
            <Logo />
          </Link>
          <nav className="flex flex-1 items-center gap-1">
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to + qs}
                end={end}
                className={({ isActive }) =>
                  cn(
                    "flex h-16 items-center gap-2 border-b-2 border-transparent px-3 text-xs font-semibold uppercase tracking-widest text-ink-foreground/60 transition-colors hover:text-ink-foreground",
                    isActive && "border-signal text-ink-foreground",
                  )
                }
              >
                <Icon className="size-4" />
                <span className="hidden sm:inline">{label}</span>
              </NavLink>
            ))}
          </nav>
          <DropdownMenu>
            <DropdownMenuTrigger className="outline-none" aria-label="Account">
              <Avatar className="size-8">
                <AvatarFallback className="bg-signal text-xs font-semibold uppercase text-signal-foreground">
                  {username.slice(0, 2)}
                </AvatarFallback>
              </Avatar>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel>{username}</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => logout.mutate()}>
                <IconLogout /> Sign out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <Outlet />
      </main>
    </div>
  );
}
