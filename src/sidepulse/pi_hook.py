"""Adapt pi extension events to SidePulse hook events.

pi has no hook configuration. ``sidepulse agent-monitor install pi`` writes a pi extension
(``resources/pi_extension.ts``) that forwards pi lifecycle events to the shared hook
entry point, tagged with their pi name in ``pi_event``. This module maps them onto
the event vocabulary the collector already understands.
"""

from __future__ import annotations

from typing import Any

PI_WAITING_MESSAGE = "pi is waiting for your input"


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a SidePulse hook payload, or None when the event changes nothing."""
    event = payload.get("pi_event")
    if not isinstance(event, str):
        return None
    idle = payload.get("idle") is True

    normalized = dict(payload)
    if payload.get("session_file") and "transcript_path" not in normalized:
        normalized["transcript_path"] = payload["session_file"]

    if event == "session_start":
        normalized["hook_event_name"] = "SessionStart"
        normalized["source"] = payload.get("reason")
    elif event == "before_agent_start":
        normalized["hook_event_name"] = "UserPromptSubmit"
    elif event == "tool_execution_start":
        normalized["hook_event_name"] = "PreToolUse"
        normalized["tool_input"] = _dict_or_empty(payload.get("args"))
    elif event == "tool_execution_end":
        response = {"is_error": payload.get("is_error") is True}
        if isinstance(payload.get("duration_ms"), (int, float)):
            response["duration_ms"] = payload["duration_ms"]
        normalized["tool_response"] = response
        if payload.get("is_error") is True:
            normalized["hook_event_name"] = "PostToolUseFailure"
            normalized["error_details"] = _text(payload.get("error_text")) or (
                f"{payload.get('tool_name') or 'tool'} failed"
            )
        else:
            normalized["hook_event_name"] = "PostToolUse"
    elif event == "ui_prompt_start":
        # An extension dialog is blocking pi on the user. No tool_input: the collector
        # would otherwise treat it as a pending tool permission and hold the state.
        normalized["hook_event_name"] = "PermissionRequest"
        title = _text(payload.get("title"))
        normalized["message"] = f"{PI_WAITING_MESSAGE}: {title}" if title else PI_WAITING_MESSAGE
    elif event == "ui_prompt_end":
        normalized["hook_event_name"] = "Notification"
        normalized["sidepulse_status"] = "ready" if idle else "working"
    elif event == "session_before_compact":
        normalized["hook_event_name"] = "PreCompact"
        normalized["trigger"] = payload.get("reason")
    elif event == "session_compact":
        normalized["hook_event_name"] = "PostCompact"
        normalized["trigger"] = payload.get("reason")
        if idle:
            # A manual /compact while idle must not leave the agent shown as working.
            normalized["sidepulse_status"] = "ready"
    elif event == "session_compact_failed":
        if not idle:
            return None  # The run carries on and reports its own outcome.
        normalized["hook_event_name"] = "PostCompact"
        normalized["trigger"] = payload.get("reason")
        if payload.get("aborted") is True:
            normalized["sidepulse_status"] = "ready"
        else:
            normalized["sidepulse_status"] = "error"
            normalized["message"] = _text(payload.get("error_message")) or "pi compaction failed"
    elif event == "agent_settled":
        stop_reason = payload.get("stop_reason")
        if payload.get("aborted") is True or stop_reason == "aborted":
            normalized["hook_event_name"] = "Interrupt"
        elif stop_reason == "error":
            normalized["hook_event_name"] = "StopFailure"
            normalized["error_details"] = _text(payload.get("error_message")) or "pi run failed"
        else:
            normalized["hook_event_name"] = "Stop"
            normalized["last_assistant_message"] = _text(payload.get("assistant_text")) or ""
    elif event == "session_shutdown":
        normalized["hook_event_name"] = "SessionEnd"
    else:
        return None

    return normalized


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
