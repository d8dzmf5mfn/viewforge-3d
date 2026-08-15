import type { AnnotationPackage, CliHealth, CliJob } from './types'

async function checkedJson<T>(response: Response): Promise<T> {
  const body = await response.json() as T & { error?: string }
  if (!response.ok) throw new Error(body.error ?? `HTTP ${response.status}`)
  return body
}

export async function fetchCliHealth(signal?: AbortSignal): Promise<CliHealth> {
  return checkedJson<CliHealth>(await fetch('/api/annotation/health', { signal }))
}

export async function createCliJob(
  annotationPackage: AnnotationPackage,
  screenshotDataUrl: string,
): Promise<CliJob> {
  return checkedJson<CliJob>(await fetch('/api/annotation/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ annotationPackage, screenshotDataUrl }),
  }))
}

export async function fetchCliJob(id: string, signal?: AbortSignal): Promise<CliJob> {
  return checkedJson<CliJob>(await fetch(`/api/annotation/jobs/${encodeURIComponent(id)}`, { signal }))
}

export async function cancelCliJob(id: string): Promise<CliJob> {
  return checkedJson<CliJob>(await fetch(`/api/annotation/jobs/${encodeURIComponent(id)}/cancel`, {
    method: 'POST',
  }))
}
