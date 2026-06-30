import logging

from src.main import SecretRedactionFilter, _redact_log_message, setup_logging


def test_redact_log_message_hides_telegram_bot_token():
    text = (
        "POST https://api.telegram.org/"
        "bot1234567890:AASecret_token-Value/getUpdates"
    )

    assert _redact_log_message(text) == (
        "POST https://api.telegram.org/bot***/getUpdates"
    )


def test_secret_redaction_filter_scrubs_record_args():
    record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="HTTP %s",
        args=("https://api.telegram.org/bot123456:ABC_token/sendMessage",),
        exc_info=None,
    )

    assert SecretRedactionFilter().filter(record) is True
    assert record.getMessage() == "HTTP https://api.telegram.org/bot***/sendMessage"


def test_setup_logging_installs_redaction_filter_on_handlers():
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_filters = root_logger.filters[:]
    original_level = root_logger.level
    try:
        root_logger.handlers.clear()
        root_logger.filters.clear()

        setup_logging(debug=False)

        assert any(
            isinstance(item, SecretRedactionFilter) for item in root_logger.filters
        )
        assert root_logger.handlers
        assert any(
            isinstance(item, SecretRedactionFilter)
            for handler in root_logger.handlers
            for item in handler.filters
        )
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(original_handlers)
        root_logger.filters.clear()
        root_logger.filters.extend(original_filters)
        root_logger.setLevel(original_level)
