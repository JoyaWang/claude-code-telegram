from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import TelegramError

from src.integrations.admin_quality_outbox import (
    AdminQualityOutboxRelay,
    QualityApiResponse,
)


class FakeQualityClient:
    def __init__(self, deliveries):
        self.deliveries = deliveries
        self.calls = []

    @property
    def is_configured(self):
        return True

    async def call_action(self, action, payload):
        self.calls.append((action, payload))
        if action == "claim_notification_delivery":
            return QualityApiResponse(status_code=200, ok=True, data=self.deliveries)
        return QualityApiResponse(status_code=200, ok=True, data={"ok": True})


@pytest.mark.asyncio
async def test_outbox_relay_sends_claimed_delivery_and_marks_sent():
    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=459))
    client = FakeQualityClient(
        [
            {
                "id": "notification-1",
                "project_key": "laicai",
                "runtime_env": "prod",
                "channel": "telegram",
                "status": "pending",
                "subject": "发到哪个环境？",
                "body": "请在 admin 或 Telegram 中选择：发 dev / 发 prod / 暂不发版",
                "metadata": {
                    "decisionToken": "decision-token",
                    "options": [
                        {"key": "dev", "label": "发 dev"},
                        {"key": "prod", "label": "发 prod"},
                    ],
                },
            }
        ]
    )
    relay = AdminQualityOutboxRelay(
        bot=bot,
        client=client,
        default_chat_ids=[7277903805],
        runner_id="telegram-bot:test",
        project_key="laicai",
        runtime_env="prod",
        limit=5,
    )

    sent = await relay.poll_once()

    assert sent == 1
    bot.send_message.assert_called_once()
    message = bot.send_message.call_args.kwargs["text"]
    assert "发到哪个环境？" in message
    assert "decision:decision-token:dev" in message
    assert "decision:decision-token:prod" in message
    assert client.calls[0] == (
        "claim_notification_delivery",
        {
            "channel": "telegram",
            "runner_id": "telegram-bot:test",
            "limit": 5,
            "project_key": "laicai",
            "runtime_env": "prod",
        },
    )
    assert client.calls[1] == (
        "update_notification_delivery",
        {
            "project_key": "laicai",
            "notification_delivery_id": "notification-1",
            "status": "sent",
            "provider_message_id": "telegram:459",
            "error_message": None,
            "runner_id": "telegram-bot:test",
        },
    )


@pytest.mark.asyncio
async def test_outbox_relay_marks_failed_when_telegram_send_fails():
    bot = AsyncMock()
    bot.send_message = AsyncMock(side_effect=TelegramError("network down"))
    client = FakeQualityClient(
        [
            {
                "id": "notification-1",
                "project_key": "laicai",
                "subject": "质量闭环通知",
                "body": "hello",
                "metadata": {},
            }
        ]
    )
    relay = AdminQualityOutboxRelay(
        bot=bot,
        client=client,
        default_chat_ids=[7277903805],
        runner_id="telegram-bot:test",
    )

    sent = await relay.poll_once()

    assert sent == 0
    assert client.calls[1] == (
        "update_notification_delivery",
        {
            "project_key": "laicai",
            "notification_delivery_id": "notification-1",
            "status": "failed",
            "provider_message_id": None,
            "error_message": "network down",
            "runner_id": "telegram-bot:test",
        },
    )
