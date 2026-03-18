import { useState, useEffect, useCallback } from 'react'
import { Plus, Upload, Download, Trash2, Search, BarChart3, CheckCircle, Copy, Tag, X, ChevronDown, ChevronUp, Database } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input, Textarea } from '@/components/ui/input'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

interface Example {
  id: string
  messages?: Message[]
  text?: string
  tags: string[]
  created_at: string
}

interface Dataset {
  id: string
  name: string
  description: string
  format: string
  example_count: number
}

export default function DataEditorPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedDs, setSelectedDs] = useState<string | null>(null)
  const [examples, setExamples] = useState<Example[]>([])
  const [total, setTotal] = useState(0)
  const [search, setSearch] = useState('')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [editingId, setEditingId] = useState<string | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [showStats, setShowStats] = useState(false)
  const [stats, setStats] = useState<any>(null)
  const [validation, setValidation] = useState<any>(null)
  const [newDs, setNewDs] = useState({ name: '', description: '', format: 'chat' })
  const [expandedExamples, setExpandedExamples] = useState<Set<string>>(new Set())

  const loadDatasets = useCallback(async () => {
    const res = await api.getDatasets()
    setDatasets(res.datasets)
  }, [])

  const loadExamples = useCallback(async () => {
    if (!selectedDs) return
    const res = await api.getExamples(selectedDs, { search, limit: '200' })
    setExamples(res.examples)
    setTotal(res.total)
  }, [selectedDs, search])

  useEffect(() => { loadDatasets() }, [loadDatasets])
  useEffect(() => { loadExamples() }, [loadExamples])

  const createDataset = async () => {
    await api.createDataset(newDs)
    setNewDs({ name: '', description: '', format: 'chat' })
    setShowCreate(false)
    loadDatasets()
  }

  const deleteDataset = async (id: string) => {
    if (!confirm('Delete this dataset?')) return
    await api.deleteDataset(id)
    if (selectedDs === id) setSelectedDs(null)
    loadDatasets()
  }

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!selectedDs || !e.target.files?.[0]) return
    await api.importData(selectedDs, e.target.files[0])
    loadExamples()
    loadDatasets()
  }

  const addExample = async () => {
    if (!selectedDs) return
    const ds = datasets.find(d => d.id === selectedDs)
    if (ds?.format === 'chat') {
      await api.addExample(selectedDs, {
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: '' },
          { role: 'assistant', content: '' },
        ],
        tags: [],
      })
    } else {
      await api.addExample(selectedDs, { text: '', tags: [] })
    }
    loadExamples()
    loadDatasets()
  }

  const updateMessage = async (exampleId: string, msgIndex: number, field: string, value: string) => {
    const ex = examples.find(e => e.id === exampleId)
    if (!ex?.messages) return
    const msgs = [...ex.messages]
    msgs[msgIndex] = { ...msgs[msgIndex], [field]: value }
    await api.updateExample(selectedDs!, exampleId, { messages: msgs })
    loadExamples()
  }

  const addMessage = async (exampleId: string, role: 'user' | 'assistant') => {
    const ex = examples.find(e => e.id === exampleId)
    if (!ex?.messages) return
    const msgs = [...ex.messages, { role, content: '' }]
    await api.updateExample(selectedDs!, exampleId, { messages: msgs })
    loadExamples()
  }

  const removeMessage = async (exampleId: string, msgIndex: number) => {
    const ex = examples.find(e => e.id === exampleId)
    if (!ex?.messages) return
    const msgs = ex.messages.filter((_, i) => i !== msgIndex)
    await api.updateExample(selectedDs!, exampleId, { messages: msgs })
    loadExamples()
  }

  const deleteExample = async (exampleId: string) => {
    if (!selectedDs) return
    await api.deleteExample(selectedDs, exampleId)
    loadExamples()
    loadDatasets()
  }

  const bulkDelete = async () => {
    if (!selectedDs || selectedIds.size === 0) return
    if (!confirm(`Delete ${selectedIds.size} examples?`)) return
    await api.bulkOperation(selectedDs, { action: 'delete', example_ids: Array.from(selectedIds) })
    setSelectedIds(new Set())
    loadExamples()
    loadDatasets()
  }

  const bulkDuplicate = async () => {
    if (!selectedDs || selectedIds.size === 0) return
    await api.bulkOperation(selectedDs, { action: 'duplicate', example_ids: Array.from(selectedIds) })
    setSelectedIds(new Set())
    loadExamples()
    loadDatasets()
  }

  const loadStats = async () => {
    if (!selectedDs) return
    const s = await api.getStats(selectedDs)
    setStats(s)
    setShowStats(true)
  }

  const validateDs = async () => {
    if (!selectedDs) return
    const v = await api.validateDataset(selectedDs)
    setValidation(v)
  }

  const toggleExpand = (id: string) => {
    setExpandedExamples(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const roleColors: Record<string, string> = {
    system: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    user: 'bg-green-500/20 text-green-400 border-green-500/30',
    assistant: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  }

  return (
    <div className="flex h-full">
      {/* Dataset List */}
      <div className="w-72 border-r p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold">Datasets</h2>
          <Button size="sm" variant="ghost" onClick={() => setShowCreate(true)}>
            <Plus className="w-4 h-4" />
          </Button>
        </div>

        {showCreate && (
          <Card className="p-3 space-y-2">
            <Input placeholder="Dataset name" value={newDs.name} onChange={e => setNewDs({ ...newDs, name: e.target.value })} />
            <Input placeholder="Description" value={newDs.description} onChange={e => setNewDs({ ...newDs, description: e.target.value })} />
            <select
              className="w-full rounded-md border bg-background px-3 py-2 text-sm"
              value={newDs.format}
              onChange={e => setNewDs({ ...newDs, format: e.target.value })}
            >
              <option value="chat">Chat format</option>
              <option value="raw">Raw text</option>
            </select>
            <div className="flex gap-2">
              <Button size="sm" onClick={createDataset} disabled={!newDs.name}>Create</Button>
              <Button size="sm" variant="ghost" onClick={() => setShowCreate(false)}>Cancel</Button>
            </div>
          </Card>
        )}

        <div className="flex-1 space-y-1 overflow-auto">
          {datasets.map(ds => (
            <div
              key={ds.id}
              className={`p-3 rounded-lg cursor-pointer transition-colors group ${
                selectedDs === ds.id ? 'bg-primary/10 border border-primary/20' : 'hover:bg-accent'
              }`}
              onClick={() => setSelectedDs(ds.id)}
            >
              <div className="flex items-center justify-between">
                <span className="font-medium text-sm">{ds.name}</span>
                <button
                  className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive"
                  onClick={e => { e.stopPropagation(); deleteDataset(ds.id) }}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {ds.example_count} examples · {ds.format}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Editor */}
      <div className="flex-1 flex flex-col">
        {selectedDs ? (
          <>
            {/* Toolbar */}
            <div className="border-b p-4 flex items-center gap-3 flex-wrap">
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  className="pl-9"
                  placeholder="Search examples..."
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                />
              </div>
              <Button size="sm" onClick={addExample}><Plus className="w-4 h-4 mr-1" /> Add</Button>
              <label>
                <Button size="sm" variant="outline" asChild>
                  <span><Upload className="w-4 h-4 mr-1" /> Import</span>
                </Button>
                <input type="file" className="hidden" accept=".jsonl,.csv,.parquet" onChange={handleImport} />
              </label>
              <a href={api.exportData(selectedDs, 'jsonl')} download>
                <Button size="sm" variant="outline"><Download className="w-4 h-4 mr-1" /> Export</Button>
              </a>
              <Button size="sm" variant="outline" onClick={loadStats}><BarChart3 className="w-4 h-4 mr-1" /> Stats</Button>
              <Button size="sm" variant="outline" onClick={validateDs}><CheckCircle className="w-4 h-4 mr-1" /> Validate</Button>

              {selectedIds.size > 0 && (
                <div className="flex items-center gap-2 ml-4 pl-4 border-l">
                  <span className="text-sm text-muted-foreground">{selectedIds.size} selected</span>
                  <Button size="sm" variant="ghost" onClick={bulkDuplicate}><Copy className="w-4 h-4" /></Button>
                  <Button size="sm" variant="ghost" className="text-destructive" onClick={bulkDelete}><Trash2 className="w-4 h-4" /></Button>
                </div>
              )}
            </div>

            {/* Stats Modal */}
            {showStats && stats && (
              <div className="border-b p-4 bg-card">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold">Dataset Statistics</h3>
                  <Button size="sm" variant="ghost" onClick={() => setShowStats(false)}><X className="w-4 h-4" /></Button>
                </div>
                <div className="grid grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold">{stats.total}</div>
                    <div className="text-xs text-muted-foreground">Total Examples</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold">{stats.avg_turns?.toFixed(1)}</div>
                    <div className="text-xs text-muted-foreground">Avg Turns</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold">{stats.avg_text_length?.toFixed(0)}</div>
                    <div className="text-xs text-muted-foreground">Avg Length (chars)</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold">{Object.keys(stats.tag_counts || {}).length}</div>
                    <div className="text-xs text-muted-foreground">Unique Tags</div>
                  </div>
                </div>
                {Object.keys(stats.turn_distribution || {}).length > 0 && (
                  <div className="mt-3">
                    <div className="text-xs text-muted-foreground mb-1">Turn Distribution</div>
                    <div className="flex gap-2 flex-wrap">
                      {Object.entries(stats.turn_distribution).map(([turns, count]) => (
                        <Badge key={turns} variant="secondary">{turns} turns: {count as number}</Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Validation Results */}
            {validation && (
              <div className="border-b p-4 bg-card">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold flex items-center gap-2">
                    Validation
                    {validation.valid ? (
                      <Badge variant="success">All Clear</Badge>
                    ) : (
                      <Badge variant="destructive">{validation.issues.length} Issues</Badge>
                    )}
                  </h3>
                  <Button size="sm" variant="ghost" onClick={() => setValidation(null)}><X className="w-4 h-4" /></Button>
                </div>
                {validation.issues?.slice(0, 20).map((issue: any, i: number) => (
                  <div key={i} className={`text-sm py-1 ${issue.severity === 'error' ? 'text-red-400' : 'text-yellow-400'}`}>
                    [{issue.severity}] {issue.example_id}: {issue.message}
                  </div>
                ))}
              </div>
            )}

            {/* Examples */}
            <div className="flex-1 overflow-auto p-4 space-y-3">
              <div className="text-sm text-muted-foreground mb-2">{total} examples</div>
              {examples.map((ex, idx) => (
                <Card key={ex.id} className="overflow-hidden">
                  <div className="flex items-center gap-3 p-3 border-b bg-card">
                    <input
                      type="checkbox"
                      checked={selectedIds.has(ex.id)}
                      onChange={() => toggleSelect(ex.id)}
                      className="rounded"
                    />
                    <span className="text-xs text-muted-foreground font-mono">#{idx + 1}</span>
                    <div className="flex gap-1 flex-1">
                      {ex.tags?.map(t => (
                        <Badge key={t} variant="secondary" className="text-xs">{t}</Badge>
                      ))}
                    </div>
                    <Button size="sm" variant="ghost" onClick={() => toggleExpand(ex.id)}>
                      {expandedExamples.has(ex.id) ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </Button>
                    <Button size="sm" variant="ghost" className="text-destructive" onClick={() => deleteExample(ex.id)}>
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>

                  {/* Collapsed preview */}
                  {!expandedExamples.has(ex.id) && (
                    <div className="p-3 text-sm text-muted-foreground cursor-pointer" onClick={() => toggleExpand(ex.id)}>
                      {ex.messages ? (
                        <span>{ex.messages.map(m => `[${m.role}] ${m.content.slice(0, 60)}`).join(' → ')}</span>
                      ) : (
                        <span>{ex.text?.slice(0, 200)}...</span>
                      )}
                    </div>
                  )}

                  {/* Expanded editor */}
                  {expandedExamples.has(ex.id) && (
                    <div className="p-3 space-y-2">
                      {ex.messages?.map((msg, mi) => (
                        <div key={mi} className="flex gap-2 items-start">
                          <div className={`px-2 py-1 rounded text-xs font-medium border min-w-[70px] text-center ${roleColors[msg.role] || ''}`}>
                            {msg.role}
                          </div>
                          <Textarea
                            className="flex-1 min-h-[60px]"
                            value={msg.content}
                            onChange={e => updateMessage(ex.id, mi, 'content', e.target.value)}
                          />
                          <Button size="icon" variant="ghost" className="text-muted-foreground" onClick={() => removeMessage(ex.id, mi)}>
                            <X className="w-3.5 h-3.5" />
                          </Button>
                        </div>
                      ))}
                      {ex.text !== undefined && (
                        <Textarea
                          className="w-full min-h-[120px]"
                          value={ex.text || ''}
                          onChange={async e => {
                            await api.updateExample(selectedDs, ex.id, { text: e.target.value })
                            loadExamples()
                          }}
                        />
                      )}
                      {ex.messages && (
                        <div className="flex gap-2 pt-2">
                          <Button size="sm" variant="outline" onClick={() => addMessage(ex.id, 'user')}>+ User</Button>
                          <Button size="sm" variant="outline" onClick={() => addMessage(ex.id, 'assistant')}>+ Assistant</Button>
                        </div>
                      )}
                    </div>
                  )}
                </Card>
              ))}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg">Select or create a dataset to get started</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
