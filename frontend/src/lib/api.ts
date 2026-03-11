const BASE = '/api'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

export const api = {
  // Models
  getModels: () => request<{ models: any[] }>('/models'),
  
  // Datasets
  getDatasets: () => request<{ datasets: any[] }>('/data/datasets'),
  createDataset: (data: any) => request<any>('/data/datasets', { method: 'POST', body: JSON.stringify(data) }),
  getDataset: (id: string) => request<any>(`/data/datasets/${id}`),
  deleteDataset: (id: string) => request<any>(`/data/datasets/${id}`, { method: 'DELETE' }),
  getExamples: (id: string, params?: Record<string, any>) => {
    const qs = new URLSearchParams(params).toString()
    return request<any>(`/data/datasets/${id}/examples?${qs}`)
  },
  addExample: (id: string, data: any) => request<any>(`/data/datasets/${id}/examples`, { method: 'POST', body: JSON.stringify(data) }),
  updateExample: (dsId: string, exId: string, data: any) => request<any>(`/data/datasets/${dsId}/examples/${exId}`, { method: 'PUT', body: JSON.stringify(data) }),
  deleteExample: (dsId: string, exId: string) => request<any>(`/data/datasets/${dsId}/examples/${exId}`, { method: 'DELETE' }),
  bulkOperation: (id: string, data: any) => request<any>(`/data/datasets/${id}/bulk`, { method: 'POST', body: JSON.stringify(data) }),
  importData: async (id: string, file: File) => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${BASE}/data/datasets/${id}/import`, { method: 'POST', body: form })
    return res.json()
  },
  exportData: (id: string, format: string) => `${BASE}/data/datasets/${id}/export?format=${format}`,
  getStats: (id: string) => request<any>(`/data/datasets/${id}/stats`),
  validateDataset: (id: string) => request<any>(`/data/datasets/${id}/validate`, { method: 'POST' }),

  // Training
  startTraining: (config: any) => request<{ job_id: string }>('/training/start', { method: 'POST', body: JSON.stringify(config) }),
  getJobs: () => request<{ jobs: any[] }>('/training/jobs'),
  getJob: (id: string) => request<any>(`/training/jobs/${id}`),
  stopJob: (id: string) => request<any>(`/training/jobs/${id}/stop`, { method: 'POST' }),
  deleteJob: (id: string) => request<any>(`/training/jobs/${id}`, { method: 'DELETE' }),
  exportModel: (id: string, merge: boolean) => request<any>(`/training/jobs/${id}/export`, { method: 'POST', body: JSON.stringify({ merge }) }),
  getGpuStatus: () => request<any>('/training/gpu'),

  // Inference
  generate: (data: any) => request<any>('/inference/generate', { method: 'POST', body: JSON.stringify(data) }),
  getAdapters: () => request<{ adapters: any[] }>('/inference/adapters'),
  saveBookmark: (data: any) => request<any>('/inference/bookmarks', { method: 'POST', body: JSON.stringify(data) }),
  getBookmarks: () => request<{ bookmarks: any[] }>('/inference/bookmarks'),
  deleteBookmark: (id: string) => request<any>(`/inference/bookmarks/${id}`, { method: 'DELETE' }),
}
