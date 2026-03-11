import { Routes, Route, NavLink } from 'react-router-dom'
import { useCallback, useState } from 'react'
import { Database, Cpu, MessageSquare, Activity } from 'lucide-react'
import { useWebSocket } from './hooks/useWebSocket'
import DataEditorPage from './pages/DataEditorPage'
import TrainingPage from './pages/TrainingPage'
import PlaygroundPage from './pages/PlaygroundPage'
import { cn } from './lib/utils'

export default function App() {
  const [wsMessages, setWsMessages] = useState<any[]>([])
  const [connected, setConnected] = useState(false)

  const handleWsMessage = useCallback((msg: any) => {
    setWsMessages(prev => [...prev.slice(-200), msg])
  }, [])

  const { connected: wsConnected } = useWebSocket(handleWsMessage)

  const navItems = [
    { to: '/', icon: Database, label: 'Data Editor' },
    { to: '/training', icon: Cpu, label: 'Training' },
    { to: '/playground', icon: MessageSquare, label: 'Playground' },
  ]

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <nav className="w-64 border-r bg-card flex flex-col">
        <div className="p-6 border-b">
          <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            Qwen Fine-Tuner
          </h1>
          <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
            <div className={cn('w-2 h-2 rounded-full', wsConnected ? 'bg-emerald-400' : 'bg-red-400')} />
            {wsConnected ? 'Connected' : 'Disconnected'}
          </div>
        </div>

        <div className="flex-1 p-3 space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-primary/10 text-primary'
                    : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                )
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </div>

        <div className="p-4 border-t text-xs text-muted-foreground">
          Qwen3 Fine-Tuning Toolkit
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<DataEditorPage />} />
          <Route path="/training" element={<TrainingPage wsMessages={wsMessages} />} />
          <Route path="/playground" element={<PlaygroundPage />} />
        </Routes>
      </main>
    </div>
  )
}
