import { useState, useEffect, useRef } from 'react'
import { Send, Bookmark, Settings2, Plus, Trash2, Copy, Columns, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input, Textarea } from '@/components/ui/input'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

interface ChatPanel {
  id: string
  model_id: string
  adapter_path: string | null
  messages: Message[]
  loading: boolean
}

export default function PlaygroundPage() {
  const [models, setModels] = useState<any[]>([])
  const [adapters, setAdapters] = useState<any[]>([])
  const [bookmarks, setBookmarks] = useState<any[]>([])
  const [showBookmarks, setShowBookmarks] = useState(false)
  const [sideBySide, setSideBySide] = useState(false)

  // Generation params
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState(0.9)
  const [topK, setTopK] = useState(50)
  const [maxTokens, setMaxTokens] = useState(512)

  const [panels, setPanels] = useState<ChatPanel[]>([
    { id: '1', model_id: 'Qwen/Qwen3-0.6B', adapter_path: null, messages: [], loading: false },
  ])

  const [input, setInput] = useState('')
  const chatEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.getModels().then(r => setModels(r.models))
    api.getAdapters().then(r => setAdapters(r.adapters)).catch(() => {})
    api.getBookmarks().then(r => setBookmarks(r.bookmarks)).catch(() => {})
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [panels])

  const toggleSideBySide = () => {
    if (!sideBySide) {
      setPanels(prev => {
        if (prev.length < 2) {
          return [...prev, { id: '2', model_id: prev[0].model_id, adapter_path: null, messages: [...prev[0].messages], loading: false }]
        }
        return prev
      })
    } else {
      setPanels(prev => prev.slice(0, 1))
    }
    setSideBySide(!sideBySide)
  }

  const sendMessage = async () => {
    if (!input.trim()) return
    const userMsg: Message = { role: 'user', content: input.trim() }
    setInput('')

    setPanels(prev => prev.map(p => ({
      ...p,
      messages: [...p.messages, userMsg],
      loading: true,
    })))

    // Send to all panels concurrently
    const promises = panels.map(async (panel) => {
      try {
        const msgs = [...panel.messages, userMsg]
        const res = await api.generate({
          model_id: panel.model_id,
          adapter_path: panel.adapter_path,
          messages: msgs,
          temperature,
          top_p: topP,
          top_k: topK,
          max_tokens: maxTokens,
        })
        return { panelId: panel.id, response: res.response }
      } catch (err: any) {
        return { panelId: panel.id, response: `Error: ${err.message}` }
      }
    })

    const results = await Promise.all(promises)

    setPanels(prev => prev.map(p => {
      const result = results.find(r => r.panelId === p.id)
      if (result) {
        return {
          ...p,
          messages: [...p.messages.filter(m => !(m === userMsg)), userMsg, { role: 'assistant' as const, content: result.response }],
          loading: false,
        }
      }
      return p
    }))
  }

  const clearChat = (panelId: string) => {
    setPanels(prev => prev.map(p => p.id === panelId ? { ...p, messages: [] } : p))
  }

  const updatePanel = (panelId: string, field: string, value: any) => {
    setPanels(prev => prev.map(p => p.id === panelId ? { ...p, [field]: value } : p))
  }

  const saveBookmark = async (panelId: string) => {
    const panel = panels.find(p => p.id === panelId)
    if (!panel || panel.messages.length === 0) return
    const lastAssistant = [...panel.messages].reverse().find(m => m.role === 'assistant')
    if (!lastAssistant) return
    const bm = await api.saveBookmark({
      model_id: panel.model_id,
      adapter_path: panel.adapter_path,
      messages: panel.messages,
      response: lastAssistant.content,
      generation_params: { temperature, top_p: topP, top_k: topK, max_tokens: maxTokens },
    })
    setBookmarks(prev => [bm, ...prev])
  }

  const deleteBookmark = async (id: string) => {
    await api.deleteBookmark(id)
    setBookmarks(prev => prev.filter(b => b.id !== id))
  }

  const roleColors: Record<string, string> = {
    system: 'text-blue-400',
    user: 'text-emerald-400',
    assistant: 'text-purple-400',
  }

  const ChatPanelView = ({ panel }: { panel: ChatPanel }) => (
    <div className="flex-1 flex flex-col min-w-0">
      {/* Panel Header */}
      <div className="border-b p-3 flex items-center gap-3">
        <select
          className="rounded-md border bg-background px-3 py-1.5 text-sm flex-1"
          value={panel.model_id}
          onChange={e => updatePanel(panel.id, 'model_id', e.target.value)}
        >
          {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
        </select>
        <select
          className="rounded-md border bg-background px-3 py-1.5 text-sm"
          value={panel.adapter_path || ''}
          onChange={e => updatePanel(panel.id, 'adapter_path', e.target.value || null)}
        >
          <option value="">No adapter (base)</option>
          {adapters.map(a => <option key={a.path} value={a.path}>{a.name}</option>)}
        </select>
        <Button size="sm" variant="ghost" onClick={() => saveBookmark(panel.id)} title="Bookmark">
          <Bookmark className="w-4 h-4" />
        </Button>
        <Button size="sm" variant="ghost" onClick={() => clearChat(panel.id)} title="Clear">
          <Trash2 className="w-4 h-4" />
        </Button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {panel.messages.length === 0 && (
          <div className="text-center text-muted-foreground py-12">
            <p className="text-lg mb-2">Start a conversation</p>
            <p className="text-sm">Type a message below to chat with {panel.model_id.split('/')[1]}</p>
          </div>
        )}
        {panel.messages.map((msg, i) => (
          <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
            <div className={`max-w-[80%] rounded-lg p-3 ${
              msg.role === 'user'
                ? 'bg-primary text-primary-foreground'
                : msg.role === 'assistant'
                  ? 'bg-card border'
                  : 'bg-blue-500/10 border border-blue-500/20'
            }`}>
              <div className={`text-xs font-medium mb-1 ${roleColors[msg.role]}`}>{msg.role}</div>
              <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
            </div>
          </div>
        ))}
        {panel.loading && (
          <div className="flex gap-3">
            <div className="bg-card border rounded-lg p-3">
              <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>
    </div>
  )

  return (
    <div className="flex h-full">
      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        <div className="flex flex-1 min-h-0">
          {panels.map(p => (
            <ChatPanelView key={p.id} panel={p} />
          ))}
        </div>

        {/* Input */}
        <div className="border-t p-4">
          <div className="flex gap-2 max-w-4xl mx-auto">
            <Textarea
              className="flex-1 min-h-[44px] max-h-32 resize-none"
              placeholder="Type a message..."
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  sendMessage()
                }
              }}
            />
            <Button onClick={sendMessage} disabled={panels.some(p => p.loading)}>
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Settings Sidebar */}
      <div className="w-72 border-l p-4 space-y-4 overflow-auto">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold flex items-center gap-2"><Settings2 className="w-4 h-4" /> Generation</h3>
          <Button size="sm" variant={sideBySide ? 'default' : 'outline'} onClick={toggleSideBySide}>
            <Columns className="w-4 h-4 mr-1" /> Compare
          </Button>
        </div>

        <div className="space-y-3">
          <div>
            <label className="text-xs text-muted-foreground flex justify-between">
              Temperature <span>{temperature}</span>
            </label>
            <input type="range" min="0" max="2" step="0.05" value={temperature} onChange={e => setTemperature(Number(e.target.value))}
              className="w-full accent-primary" />
          </div>
          <div>
            <label className="text-xs text-muted-foreground flex justify-between">
              Top P <span>{topP}</span>
            </label>
            <input type="range" min="0" max="1" step="0.05" value={topP} onChange={e => setTopP(Number(e.target.value))}
              className="w-full accent-primary" />
          </div>
          <div>
            <label className="text-xs text-muted-foreground flex justify-between">
              Top K <span>{topK}</span>
            </label>
            <input type="range" min="0" max="200" step="1" value={topK} onChange={e => setTopK(Number(e.target.value))}
              className="w-full accent-primary" />
          </div>
          <div>
            <label className="text-xs text-muted-foreground flex justify-between">
              Max Tokens <span>{maxTokens}</span>
            </label>
            <input type="range" min="64" max="4096" step="64" value={maxTokens} onChange={e => setMaxTokens(Number(e.target.value))}
              className="w-full accent-primary" />
          </div>
        </div>

        {/* Bookmarks */}
        <div>
          <button
            className="text-sm font-semibold flex items-center gap-2 w-full py-2"
            onClick={() => setShowBookmarks(!showBookmarks)}
          >
            <Bookmark className="w-4 h-4" /> Bookmarks ({bookmarks.length})
          </button>
          {showBookmarks && (
            <div className="space-y-2 mt-2">
              {bookmarks.map(bm => (
                <Card key={bm.id} className="p-2">
                  <div className="text-xs text-muted-foreground mb-1">{bm.model_id?.split('/')[1]}</div>
                  <div className="text-xs truncate">{bm.response?.slice(0, 100)}...</div>
                  <div className="flex justify-end mt-1">
                    <Button size="sm" variant="ghost" onClick={() => deleteBookmark(bm.id)}>
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </Card>
              ))}
              {bookmarks.length === 0 && <p className="text-xs text-muted-foreground">No bookmarks yet</p>}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
