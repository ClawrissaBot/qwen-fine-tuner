"""WebSocket connection manager for real-time updates."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket


class WSManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_sync(self, data: dict):
        """Fire-and-forget broadcast from sync code."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast(data))
        except RuntimeError:
            pass


ws_manager = WSManager()
