from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class ExpoPushMessage:
    to: str
    title: str
    body: str
    data: dict[str, Any] | None = None
    sound: bool | str | None = True
    channel_id: str | None = "alerts"


async def send_expo_push(*, messages: list[ExpoPushMessage]) -> dict[str, Any]:
    """
    Send push notifications via Expo.

    Notes:
    - Does not require auth for basic push.
    - You must provide valid Expo push tokens (ExponentPushToken[...]).
    """
    if not messages:
        return {"ok": True, "sent": 0, "response": None}

    url = "https://exp.host/--/api/v2/push/send"
    payload = [
        {
            "to": m.to,
            "title": m.title,
            "body": m.body,
            "data": (m.data or {}),
            "sound": m.sound,
            # Android: must match channel created in app (`alerts`)
            "channelId": m.channel_id,
        }
        for m in messages
    ]

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=payload)
        # Expo returns 200 even for per-message errors; keep full JSON for debugging.
        try:
            data = resp.json()
        except Exception:
            data = {"text": resp.text}

    return {"ok": resp.status_code == 200, "sent": len(messages), "response": data}
