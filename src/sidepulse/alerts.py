"""Emergency alerts that stay active until the user dismisses them.

Anything can raise one: ``sidepulse alert raise "message"`` from a script, a cron job,
or an agent hook (with no message it reads the hook's JSON payload from stdin). Alerts
are stored in ``alerts.json`` in the SidePulse state directory, so they survive
restarts, and the running menu-bar app is told to refresh at once.

While any alert is active, every LED display SidePulse drives pulses red, overriding
agent, battery, lid, and preview animations, and the alerts are listed at the top of
the SidePulse menu. Dismiss them from the menu or with ``sidepulse alert dismiss``.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .ipc import notify_alerts_changed
from .led_status import apply_brightness
from .providers import candidate_state_dirs, default_state_dir

ALERTS_FILE_NAME = "alerts.json"
ALERTS_FORMAT_VERSION = 1
DEFAULT_ALERT_MESSAGE = "Attention needed"
MAX_MESSAGE_CHARS = 500
MAX_SOURCE_CHARS = 80
# Repeats of the same alert collapse into one entry, so this only caps unique alerts.
MAX_ACTIVE_ALERTS = 100
# A fast red pulse: one beat a second, unlike the 1.6 s amber "needs input" pulse.
ALERT_PROGRAM = "off\n#FF0000 800ms pulse\noff 200ms none\nrepeat"


@dataclass(frozen=True)
class Alert:
    alert_id: str
    message: str
    source: str
    key: str
    raised_at: str
    last_raised_at: str
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.alert_id,
            "message": self.message,
            "source": self.source,
            "key": self.key,
            "raised_at": self.raised_at,
            "last_raised_at": self.last_raised_at,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Alert | None":
        if not isinstance(data, dict):
            return None
        alert_id = data.get("id")
        message = data.get("message")
        raised_at = data.get("raised_at")
        if not (isinstance(alert_id, str) and alert_id and isinstance(message, str)):
            return None
        if not isinstance(raised_at, str):
            return None
        source = data.get("source") if isinstance(data.get("source"), str) else ""
        key = data.get("key") if isinstance(data.get("key"), str) else alert_id
        last = data.get("last_raised_at")
        count = data.get("count")
        return cls(
            alert_id=alert_id,
            message=message,
            source=source,
            key=key,
            raised_at=raised_at,
            last_raised_at=last if isinstance(last, str) else raised_at,
            count=count if isinstance(count, int) and count > 0 else 1,
        )


def default_alerts_path(home: Path | None = None) -> Path:
    return default_state_dir(home) / ALERTS_FILE_NAME


def candidate_alerts_paths(home: Path | None = None) -> tuple[Path, ...]:
    """Where the CLI or the app may have stored alerts (see candidate_state_dirs)."""
    return tuple(directory / ALERTS_FILE_NAME for directory in candidate_state_dirs(home))


def alert_program(brightness: int | float = 255) -> str:
    return apply_brightness(ALERT_PROGRAM, brightness)


def clean_text(value: Any, limit: int) -> str:
    """One line of text: menus show alerts on a single line."""
    if not isinstance(value, str):
        return ""
    text = re.sub(r"\s+", " ", value).strip()
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}…"


def message_from_payload(text: str) -> tuple[str, str]:
    """Return (message, source) from stdin: an agent hook's JSON payload, or plain text."""
    stripped = text.strip()
    try:
        payload = json.loads(stripped) if stripped.startswith("{") else None
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict):
        return clean_text(stripped, MAX_MESSAGE_CHARS), ""
    message = ""
    for field in ("message", "title", "reason", "error", "error_details"):
        message = clean_text(payload.get(field), MAX_MESSAGE_CHARS)
        if message:
            break
    if not message:
        message = clean_text(payload.get("hook_event_name"), MAX_MESSAGE_CHARS)
    return message, clean_text(payload.get("hook_event_name"), MAX_SOURCE_CHARS)


def alert_time_label(timestamp: str) -> str:
    """Local time for an alert's timestamp: "14:05" today, "3 Nov 14:05" otherwise."""
    try:
        moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return timestamp
    if moment.date() == datetime.now().astimezone().date():
        return moment.strftime("%H:%M")
    return f"{moment.day} {moment.strftime('%b %H:%M')}"


def load_alerts(paths: tuple[Path, ...] | None = None) -> list[Alert]:
    """Active alerts from every candidate store, oldest first."""
    seen: dict[str, Alert] = {}
    for path in paths if paths is not None else candidate_alerts_paths():
        for alert in _read_alerts(path):
            seen.setdefault(alert.alert_id, alert)
    # Stable sort: alerts raised in the same second keep the order they were raised in.
    return sorted(seen.values(), key=lambda alert: alert.raised_at)


def raise_alert(
    message: str,
    *,
    source: str = "",
    key: str | None = None,
    path: Path | None = None,
    now: datetime | None = None,
    notify: bool = True,
) -> Alert:
    """Record an alert, or bump the count of the active alert with the same key."""
    text = clean_text(message, MAX_MESSAGE_CHARS) or DEFAULT_ALERT_MESSAGE
    origin = clean_text(source, MAX_SOURCE_CHARS)
    alert_key = key.strip() if isinstance(key, str) and key.strip() else f"{origin}\n{text}"
    timestamp = _timestamp(now)
    target = (path or default_alerts_path()).expanduser()
    with _locked(target):
        alerts = _read_alerts(target)
        existing = next((alert for alert in alerts if alert.key == alert_key), None)
        if existing is not None:
            alert = replace(
                existing,
                message=text,
                source=origin,
                last_raised_at=timestamp,
                count=existing.count + 1,
            )
            alerts = [alert if item.alert_id == existing.alert_id else item for item in alerts]
        else:
            alert = Alert(
                alert_id=secrets.token_hex(4),
                message=text,
                source=origin,
                key=alert_key,
                raised_at=timestamp,
                last_raised_at=timestamp,
            )
            alerts = [*alerts, alert][-MAX_ACTIVE_ALERTS:]
        _write_alerts(target, alerts)
    if notify:
        notify_alerts_changed()
    return alert


def dismiss_alert(
    alert_id: str,
    *,
    paths: tuple[Path, ...] | None = None,
    notify: bool = True,
) -> bool:
    removed = _remove(lambda alert: alert.alert_id == alert_id, paths)
    if removed and notify:
        notify_alerts_changed()
    return removed > 0


def dismiss_all_alerts(
    *,
    paths: tuple[Path, ...] | None = None,
    notify: bool = True,
) -> int:
    removed = _remove(lambda _alert: True, paths)
    if removed and notify:
        notify_alerts_changed()
    return removed


def _remove(matches, paths: tuple[Path, ...] | None) -> int:
    removed = 0
    for path in paths if paths is not None else candidate_alerts_paths():
        target = path.expanduser()
        if not target.exists():
            continue
        with _locked(target):
            alerts = _read_alerts(target)
            kept = [alert for alert in alerts if not matches(alert)]
            if len(kept) != len(alerts):
                _write_alerts(target, kept)
                removed += len(alerts) - len(kept)
    return removed


def _timestamp(now: datetime | None) -> str:
    moment = now or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_alerts(path: Path) -> list[Alert]:
    try:
        data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    raw = data.get("alerts") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    return [alert for alert in (Alert.from_dict(item) for item in raw) if alert is not None]


def _write_alerts(path: Path, alerts: list[Alert]) -> None:
    document = {
        "version": ALERTS_FORMAT_VERSION,
        "alerts": [alert.to_dict() for alert in alerts],
    }
    handle, temporary = tempfile.mkstemp(prefix=".alerts-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Serialise read-modify-write between the CLI, the app, and the service."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(f"{path.name}.lock"), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
