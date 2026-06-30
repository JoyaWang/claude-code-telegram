"""Relay admin-platform Telegram notification outbox through this bot."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import structlog
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError

logger = structlog.get_logger()

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class QualityApiResponse:
    """Normalized admin-platform quality API response."""

    status_code: int
    ok: bool
    data: Any = None
    error: Optional[str] = None


class AdminQualityApiClient:
    """Small async client for admin-platform quality runner actions."""

    def __init__(
        self,
        api_url: Optional[str],
        api_key: Optional[str],
        timeout_seconds: int = 15,
    ) -> None:
        self.api_url = api_url.strip() if api_url else None
        self.api_key = api_key.strip() if api_key else None
        self.timeout_seconds = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_url and self.api_key)

    async def call_action(self, action: str, payload: JsonDict) -> QualityApiResponse:
        if not self.is_configured:
            return QualityApiResponse(
                status_code=0,
                ok=False,
                error="ADMIN_QUALITY_API_URL or ADMIN_QUALITY_API_KEY is not configured",
            )
        body = {"action": action, **payload}
        return await asyncio.to_thread(self._post_json, body)

    def _post_json(self, payload: JsonDict) -> QualityApiResponse:
        request = Request(
            self.api_url or "",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key or "",
            },
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
            return QualityApiResponse(status_code=0, ok=False, error=str(exc.reason))
        except TimeoutError:
            return QualityApiResponse(
                status_code=0,
                ok=False,
                error="admin quality API request timed out",
            )

        try:
            parsed = json.loads(response_body) if response_body else {}
        except json.JSONDecodeError:
            parsed = {"error": response_body[:500]}
        data = parsed.get("data") if isinstance(parsed, dict) else None
        error = parsed.get("error") if isinstance(parsed, dict) else None
        return QualityApiResponse(
            status_code=status_code,
            ok=200 <= status_code < 300,
            data=data,
            error=error if isinstance(error, str) and error else None,
        )


class AdminQualityOutboxRelay:
    """Poll admin-platform notification deliveries and send them via Telegram."""

    def __init__(
        self,
        *,
        bot: Bot,
        client: AdminQualityApiClient,
        default_chat_ids: Iterable[int],
        runner_id: str = "telegram-bot:local",
        project_key: Optional[str] = None,
        runtime_env: Optional[str] = None,
        poll_interval_seconds: float = 10.0,
        limit: int = 10,
    ) -> None:
        self.bot = bot
        self.client = client
        self.default_chat_ids = list(default_chat_ids)
        self.runner_id = runner_id
        self.project_key = project_key
        self.runtime_env = runtime_env
        self.poll_interval_seconds = poll_interval_seconds
        self.limit = limit
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        if self._running:
            return
        if not self.client.is_configured:
            logger.warning("Admin quality outbox relay not configured")
            return
        if not self.default_chat_ids:
            logger.warning("Admin quality outbox relay has no target chat ids")
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="admin-quality-outbox-relay")
        logger.info(
            "Admin quality outbox relay started",
            project_key=self.project_key,
            runtime_env=self.runtime_env,
            poll_interval_seconds=self.poll_interval_seconds,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Admin quality outbox relay stopped")

    async def _run(self) -> None:
        while self._running:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover - defensive loop guard
                logger.error("Admin quality outbox poll failed", error=str(exc))
            await asyncio.sleep(self.poll_interval_seconds)

    async def poll_once(self) -> int:
        payload: JsonDict = {
            "channel": "telegram",
            "runner_id": self.runner_id,
            "limit": self.limit,
        }
        if self.project_key:
            payload["project_key"] = self.project_key
        if self.runtime_env:
            payload["runtime_env"] = self.runtime_env

        response = await self.client.call_action("claim_notification_delivery", payload)
        if not response.ok:
            logger.warning(
                "Admin quality outbox claim failed",
                status_code=response.status_code,
                error=response.error,
            )
            return 0

        deliveries = response.data if isinstance(response.data, list) else []
        sent_count = 0
        for delivery in deliveries:
            if isinstance(delivery, dict):
                if await self._send_delivery(delivery):
                    sent_count += 1
        return sent_count

    async def _send_delivery(self, delivery: JsonDict) -> bool:
        delivery_id = str(delivery.get("id") or "")
        project_key = str(delivery.get("project_key") or "")
        if not delivery_id or not project_key:
            return False

        chat_id = self._resolve_chat_id(delivery)
        text = self._format_delivery(delivery)
        reply_markup = self._decision_reply_markup(delivery)
        try:
            kwargs: JsonDict = {
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            }
            if reply_markup:
                kwargs["reply_markup"] = reply_markup
            message = await self.bot.send_message(
                **kwargs,
            )
        except TelegramError as exc:
            await self._mark_delivery(
                delivery,
                "failed",
                error_message=str(exc),
            )
            return False

        message_id = getattr(message, "message_id", None)
        await self._mark_delivery(
            delivery,
            "sent",
            provider_message_id=(
                f"telegram:{message_id}" if message_id is not None else None
            ),
        )
        return True

    def _resolve_chat_id(self, delivery: JsonDict) -> int:
        recipient = delivery.get("recipient")
        if isinstance(recipient, int):
            return recipient
        if isinstance(recipient, str) and recipient.strip():
            return int(recipient.strip())
        return self.default_chat_ids[0]

    def _format_delivery(self, delivery: JsonDict) -> str:
        subject = _optional_text(delivery.get("subject")) or "质量闭环通知"
        body = _optional_text(delivery.get("body")) or ""
        metadata = _metadata(delivery.get("metadata"))
        lines: List[str] = [subject]
        if body:
            lines.extend(["", body])

        token = _optional_text(metadata.get("decisionToken"))
        options = metadata.get("options")
        if token and isinstance(options, list) and options:
            lines.extend(["", "可点击按钮，或回复以下任一行继续："])
            for option in options:
                if not isinstance(option, dict):
                    continue
                key = _optional_text(option.get("key"))
                if not key:
                    continue
                label = _optional_text(option.get("label")) or key
                lines.append(f"decision:{token}:{key}  - {label}")
        return "\n".join(lines).strip()

    def _decision_reply_markup(
        self, delivery: JsonDict
    ) -> Optional[InlineKeyboardMarkup]:
        metadata = _metadata(delivery.get("metadata"))
        token = _optional_text(metadata.get("decisionToken"))
        options = metadata.get("options")
        if not token or not isinstance(options, list):
            return None

        rows: List[List[InlineKeyboardButton]] = []
        current_row: List[InlineKeyboardButton] = []
        for option in options:
            if not isinstance(option, dict):
                continue
            key = _optional_text(option.get("key"))
            if not key:
                continue
            label = _optional_text(option.get("label")) or key
            current_row.append(
                InlineKeyboardButton(label, callback_data=f"decision:{token}:{key}")
            )
            if len(current_row) == 2:
                rows.append(current_row)
                current_row = []
        if current_row:
            rows.append(current_row)
        return InlineKeyboardMarkup(rows) if rows else None

    async def _mark_delivery(
        self,
        delivery: JsonDict,
        status: str,
        *,
        provider_message_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        await self.client.call_action(
            "update_notification_delivery",
            {
                "project_key": delivery.get("project_key"),
                "notification_delivery_id": delivery.get("id"),
                "status": status,
                "provider_message_id": provider_message_id,
                "error_message": error_message,
                "runner_id": self.runner_id,
            },
        )


def _metadata(value: Any) -> JsonDict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _optional_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
