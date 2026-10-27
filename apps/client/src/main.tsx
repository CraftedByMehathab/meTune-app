import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, Navigate, RouterProvider, useParams } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { Toaster } from "@/components/ui/sonner";
import { ApiError } from "@/lib/api";
import { LoginPage } from "@/pages/LoginPage";
import { ReportsPage } from "@/pages/ReportsPage";
import { ResultsPage } from "@/pages/ResultsPage";
import { UploadPage } from "@/pages/UploadPage";
import "./index.css";

// Follow the OS theme: the shadcn preset keys dark mode off a `.dark` class.
const dark = window.matchMedia("(prefers-color-scheme: dark)");
const applyTheme = () => document.documentElement.classList.toggle("dark", dark.matches);
applyTheme();
dark.addEventListener("change", applyTheme);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (count, err) => !(err instanceof ApiError && err.status < 500) && count < 2,
      refetchOnWindowFocus: false,
    },
  },
});

/** Old /batches/:id links land on the batch in the upload flow. */
function LegacyBatchRedirect() {
  const { id = "" } = useParams();
  return <Navigate to={`/dashboard?batch=${encodeURIComponent(id)}`} replace />;
}

const router = createBrowserRouter([
  { path: "/login", element: <LoginPage /> },
  { path: "/", element: <Navigate to="/dashboard" replace /> },
  { path: "/batches/:id", element: <LegacyBatchRedirect /> },
  {
    path: "/dashboard",
    element: <Layout />,
    children: [
      { index: true, element: <UploadPage /> },
      { path: "results", element: <ResultsPage /> },
      { path: "reports", element: <ReportsPage /> },
    ],
  },
  { path: "*", element: <Navigate to="/dashboard" replace /> },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      <Toaster position="bottom-right" />
    </QueryClientProvider>
  </StrictMode>,
);
