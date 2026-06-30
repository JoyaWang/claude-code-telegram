"""Admin Platform quality-loop webhook client."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from telegram import Update


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class AdminQualityResult:
    """Normalized response from admin-platform quality Telegram webhook."""

    status_code: int
    ok: bool
    reply: Optional[str]
    delivery_sent: bool
    reply_markup: Optional[JsonDict] = None
    error: Optional[str] = None


class AdminQualityClient:
    """Small async wrapper around admin-platform's Telegram webhook endpoint."""

    def __init__(
        self,
        webhook_url: Optional[str],
        webhook_secret: Optional[str] = None,
        timeout_seconds: int = 15,
    ) -> None:
        self.webhook_url = webhook_url.strip() if webhook_url else None
        self.webhook_secret = webhook_secret.strip() if webhook_secret else None
        self.timeout_seconds = timeout_seconds

    @property
    def is_configured(self) -> bool:
        """Return whether outbound admin-platform forwarding is enabled."""
        return bool(self.webhook_url)

    async def forward_update(self, update: Update) -> AdminQualityResult:
        """Forward a Telegram update using admin-platform's expected shape."""
        if not self.webhook_url:
            return AdminQualityResult(
                status_code=0,
                ok=False,
                reply=None,
                delivery_sent=False,
                reply_markup=None,
                error="ADMIN_QUALITY_WEBHOOK_URL is not configured",
            )

        payload = self._payload_from_update(update)
        return await asyncio.to_thread(self._post_json, payload)

    def _post_json(self, payload: JsonDict) -> AdminQualityResult:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.webhook_secret:
            headers["x-telegram-bot-api-secret-token"] = self.webhook_secret

        request = Request(
            self.webhook_url or "",
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status_code = int(response.status)
                response_body = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            status_code = int(exc.code)
            response_body = exc.read().decode("utf-8", errors="replace")
        except URLError as exc:
            return AdminQualityResult(
                status_code=0,
                ok=False,
                reply=None,
                delivery_sent=False,
                reply_markup=None,
                error=str(exc.reason),
            )
        except TimeoutError:
            return AdminQualityResult(
                status_code=0,
                ok=False,
                reply=None,
                delivery_sent=False,
                reply_markup=None,
                error="admin quality webhook request timed out",
            )

        return self._parse_response(status_code, response_body)

    @staticmethod
    def _parse_response(status_code: int, response_body: str) -> AdminQualityResult:
        try:
            parsed = json.loads(response_body) if response_body else {}
        except json.JSONDecodeError:
            parsed = {"error": response_body[:500]}

        data = parsed.get("data") if isinstance(parsed, dict) else None
        data = data if isinstance(data, dict) else {}
        delivery = data.get("delivery")
        delivery = delivery if isinstance(delivery, dict) else {}

        reply = data.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            reply = None

        reply_markup = data.get("reply_markup") or data.get("replyMarkup")
        if not isinstance(reply_markup, dict):
            reply_markup = None

        error = parsed.get("error") if isinstance(parsed, dict) else None
        if not isinstance(error, str) or not error.strip():
            error = None

        return AdminQualityResult(
            status_code=status_code,
            ok=200 <= status_code < 300,
            reply=reply,
            delivery_sent=bool(delivery.get("sent")),
            reply_markup=reply_markup,
            error=error,
        )

    @staticmethod
    def _payload_from_update(update: Update) -> JsonDict:
        """Create the subset of Telegram update JSON used by admin-platform."""
        callback_query = _callback_query(update)
        if callback_query:
            message = getattr(callback_query, "message", None)
            chat = getattr(message, "chat", None) if message else None
            return {
                "callback_query": {
                    "id": getattr(callback_query, "id", None),
                    "data": getattr(callback_query, "data", None),
                    "message": {
                        "message_id": getattr(message, "message_id", None),
                        "chat": {
                            "id": getattr(chat, "id", None),
                            "type": getattr(chat, "type", None),
                        },
                    },
                    "from": _user_payload(getattr(callback_query, "from_user", None)),
                }
            }

        message = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        payload: JsonDict = {}
        if message:
            payload["message"] = {
                "message_id": message.message_id,
                "text": message.text,
                "chat": {
                    "id": chat.id if chat else None,
                    "type": getattr(chat, "type", None) if chat else None,
                },
                "from": _user_payload(user),
            }
        else:
            payload["from"] = _user_payload(user)
            if chat:
                payload["chat"] = {"id": chat.id, "type": getattr(chat, "type", None)}

        return payload


def _user_payload(user: Any) -> JsonDict:
    if not user:
        return {}
    return {
        "id": getattr(user, "id", None),
        "username": getattr(user, "username", None),
        "first_name": getattr(user, "first_name", None),
    }


def _callback_query(update: Update) -> Any:
    query = getattr(update, "callback_query", None)
    if query is None:
        return None
    return query if isinstance(getattr(query, "data", None), str) else None
