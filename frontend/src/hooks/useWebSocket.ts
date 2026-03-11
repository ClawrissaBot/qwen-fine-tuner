import { useEffect, useRef, useCallback, useState } from 'react'

type WSMessage = {
  type: string
  [key: string]: any
}

export function useWebSocket(onMessage: (msg: WSMessage) => void) {
  const wsRef = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`)

    ws.onopen = () => {
      setConnected(true)
      // Heartbeat
      const interval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }))
        }
      }, 30000)
      ws.addEventListener('close', () => clearInterval(interval))
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type !== 'pong') {
          onMessage(msg)
        }
      } catch {}
    }

    ws.onclose = () => {
      setConnected(false)
      setTimeout(connect, 3000)
    }

    ws.onerror = () => ws.close()
    wsRef.current = ws
  }, [onMessage])

  useEffect(() => {
    connect()
    return () => wsRef.current?.close()
  }, [connect])

  return { connected }
}
