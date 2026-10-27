import type { components } from "./api-schema";

type S = components["schemas"];
export type Me = S["Me"];
export type BatchSummary = S["BatchSummary"];
export type BatchDetail = S["BatchDetail"];
export type FileOut = S["FileOut"];
export type Result = S["Result"];
export type ManifestReport = S["ManifestReport"];
export type Metrics = S["Metrics"];

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const isForm = init?.body instanceof FormData;
  const res = await fetch(`/api${path}`, {
    credentials: "same-origin",
    ...init,
    headers: isForm ? init?.headers : { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : res.statusText;
    throw new ApiError(res.status, detail);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const api = {
  me: () => request<Me>("/auth/me"),
  login: (username: string, password: string) =>
    request<Me>("/auth/login", { method: "POST", body: JSON.stringify({ username, password }) }),
  logout: () => request<void>("/auth/logout", { method: "POST" }),
  batches: () => request<BatchSummary[]>("/batches"),
  batch: (id: string) => request<BatchDetail>(`/batches/${id}`),
  files: (id: string) => request<FileOut[]>(`/batches/${id}/files`),
  start: (id: string) => request<BatchDetail>(`/batches/${id}/start`, { method: "POST" }),
  downloadUrl: (id: string, format: "csv" | "json") => `/api/batches/${id}/download?format=${format}`,
};

/** XHR instead of fetch: fetch has no upload progress events. */
export function uploadBatch(files: File[], name: string, onProgress: (p: number) => void): Promise<BatchDetail> {
  const form = new FormData();
  if (name) form.append("name", name);
  for (const f of files) form.append("files", f, f.webkitRelativePath || f.name);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/batches");
    xhr.withCredentials = true;
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      const body = JSON.parse(xhr.responseText || "{}");
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new ApiError(xhr.status, body.detail ?? xhr.statusText));
    };
    xhr.onerror = () => reject(new ApiError(0, "Network error during upload"));
    xhr.send(form);
  });
}
