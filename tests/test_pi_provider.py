from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from sidepulse.cli import build_parser
from sidepulse.collector import AgentMonitor, SourceSpec
from sidepulse.hook import hook_log_main
from sidepulse.install import (
    install_pi_hooks,
    pi_extension_backup_dir,
    render_pi_extension,
    uninstall_pi_hooks,
)
from sidepulse.models import AgentMode, AgentStatus, provider_label
from sidepulse.origin import ProcessInfo, is_pi_process, origin_from_processes
from sidepulse.pi_hook import normalize_payload
from sidepulse.providers import (
    PI_EVENTS,
    PI_EXTENSION_MARKER,
    default_pi_extension_path,
    detect_pi_config,
    pi_extension_hook_command,
)
from sidepulse.session_actions import session_resume_command


def pi_event(name: str, **fields) -> dict:
    payload = {
        "pi_event": name,
        "session_id": "pi-session-1234",
        "cwd": "/tmp/project",
        "pi_mode": "tui",
        "idle": False,
    }
    payload.update(fields)
    return payload


class PiInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.config = default_pi_extension_path(self.home)
        self.log = self.home / "state" / "pi.jsonl"

    def test_installer_writes_managed_extension_that_detection_reads_back(self) -> None:
        result = install_pi_hooks(
            config_path=self.config,
            log_path=self.log,
            python_executable="/venv/bin/python",
        )

        self.assertTrue(result.changed)
        self.assertEqual(self.config, self.home / ".pi" / "agent" / "extensions" / "sidepulse.ts")
        text = self.config.read_text()
        self.assertIn(PI_EXTENSION_MARKER, text)
        argv = pi_extension_hook_command(text)
        self.assertEqual(argv[0], "/venv/bin/python")
        self.assertTrue(argv[1].endswith("hook_entry.py"))
        self.assertEqual(argv[2:], ["--provider", "pi", "--log", str(self.log)])
        self.assertTrue(self.log.exists())

        detected = detect_pi_config(self.home)
        self.assertTrue(detected.hooks_enabled)
        self.assertEqual(detected.hook_events, tuple(sorted(PI_EVENTS)))
        self.assertEqual(detected.log_paths, (self.log,))

    def test_reinstall_is_idempotent(self) -> None:
        install_pi_hooks(config_path=self.config, log_path=self.log, python_executable="python3")
        second = install_pi_hooks(config_path=self.config, log_path=self.log, python_executable="python3")

        self.assertFalse(second.changed)
        self.assertIsNone(second.backup_path)

    def test_installer_backs_up_a_hand_written_extension_outside_extensions_dir(self) -> None:
        self.config.parent.mkdir(parents=True)
        self.config.write_text("export default function (pi) {}\n")

        result = install_pi_hooks(config_path=self.config, log_path=self.log, python_executable="python3")

        self.assertIsNotNone(result.backup_path)
        self.assertEqual(result.backup_path.parent, pi_extension_backup_dir(self.config))
        self.assertNotEqual(result.backup_path.parent, self.config.parent)
        self.assertEqual(result.backup_path.read_text(), "export default function (pi) {}\n")
        self.assertIn(PI_EXTENSION_MARKER, self.config.read_text())

    def test_dry_run_changes_nothing(self) -> None:
        result = install_pi_hooks(config_path=self.config, log_path=self.log, dry_run=True)

        self.assertTrue(result.changed)
        self.assertFalse(self.config.exists())
        self.assertFalse(self.log.exists())

    def test_uninstaller_removes_only_the_managed_extension(self) -> None:
        install_pi_hooks(config_path=self.config, log_path=self.log, python_executable="python3")

        removed = uninstall_pi_hooks(config_path=self.config, log_path=self.log)
        again = uninstall_pi_hooks(config_path=self.config, log_path=self.log)

        self.assertTrue(removed.changed)
        self.assertFalse(again.changed)
        self.assertFalse(self.config.exists())
        self.assertFalse(detect_pi_config(self.home).exists)

    def test_uninstaller_leaves_a_hand_written_extension(self) -> None:
        self.config.parent.mkdir(parents=True)
        self.config.write_text("export default function (pi) {}\n")

        result = uninstall_pi_hooks(config_path=self.config, log_path=self.log)

        self.assertFalse(result.changed)
        self.assertTrue(self.config.exists())
        detected = detect_pi_config(self.home)
        self.assertTrue(detected.exists)
        self.assertFalse(detected.hooks_enabled)

    def test_extension_path_honours_pi_coding_agent_dir(self) -> None:
        custom = self.home / "custom-pi"
        with patch.dict(os.environ, {"PI_CODING_AGENT_DIR": str(custom)}):
            self.assertEqual(default_pi_extension_path(), custom / "extensions" / "sidepulse.ts")
            # An explicit home ignores the override, matching default_state_dir().
            self.assertEqual(
                default_pi_extension_path(self.home),
                self.home / ".pi" / "agent" / "extensions" / "sidepulse.ts",
            )

    def test_template_subscribes_to_every_forwarded_event(self) -> None:
        text = render_pi_extension(["python3", "hook_entry.py", "--provider", "pi", "--log", "/tmp/pi.jsonl"])

        self.assertNotIn("__SIDEPULSE_HOOK_COMMAND__", text)
        for event in PI_EVENTS:
            self.assertIn(f'pi.on("{event}"', text)
        # agent_end is not final in pi; the extension only records the reply there.
        self.assertIn('pi.on("agent_end"', text)
        self.assertNotIn('send("agent_end"', text)

    def test_cli_accepts_pi_and_custom_log_path(self) -> None:
        args = build_parser().parse_args(["install", "pi", "--pi-log", "/tmp/custom-pi.jsonl"])

        self.assertEqual(args.provider, "pi")
        self.assertEqual(args.pi_log, Path("/tmp/custom-pi.jsonl"))


class PiEventMappingTests(unittest.TestCase):
    def assertMapped(self, payload: dict, event_name: str, **expected) -> dict:
        normalized = normalize_payload(payload)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["hook_event_name"], event_name)
        for key, value in expected.items():
            self.assertEqual(normalized.get(key), value, key)
        return normalized

    def test_session_and_prompt_events(self) -> None:
        self.assertMapped(pi_event("session_start", reason="startup"), "SessionStart", source="startup")
        self.assertMapped(
            pi_event("before_agent_start", message_type="user", prompt="fix the test"),
            "UserPromptSubmit",
            prompt="fix the test",
        )
        self.assertMapped(pi_event("session_shutdown", reason="quit"), "SessionEnd")

    def test_tool_events(self) -> None:
        self.assertMapped(
            pi_event("tool_execution_start", tool_name="bash", args={"command": "pytest"}),
            "PreToolUse",
            tool_name="bash",
            tool_input={"command": "pytest"},
        )
        self.assertMapped(
            pi_event("tool_execution_end", tool_name="read", is_error=False, duration_ms=4),
            "PostToolUse",
            tool_response={"is_error": False, "duration_ms": 4},
        )
        self.assertMapped(
            pi_event("tool_execution_end", tool_name="bash", is_error=True, error_text="exit code 1"),
            "PostToolUseFailure",
            error_details="exit code 1",
        )
        self.assertMapped(
            pi_event("tool_execution_end", tool_name="bash", is_error=True),
            "PostToolUseFailure",
            error_details="bash failed",
        )

    def test_ui_prompts_wait_for_input_without_a_pending_tool_permission(self) -> None:
        start = self.assertMapped(
            pi_event("ui_prompt_start", kind="confirm", title="Allow rm -rf build?"),
            "PermissionRequest",
            message="pi is waiting for your input: Allow rm -rf build?",
        )
        self.assertNotIn("tool_input", start)
        self.assertMapped(pi_event("ui_prompt_end", idle=False), "Notification", sidepulse_status="working")
        self.assertMapped(pi_event("ui_prompt_end", idle=True), "Notification", sidepulse_status="ready")

    def test_compaction_events(self) -> None:
        self.assertMapped(pi_event("session_before_compact", reason="manual"), "PreCompact", trigger="manual")
        busy = self.assertMapped(pi_event("session_compact", reason="threshold"), "PostCompact")
        self.assertNotIn("sidepulse_status", busy)
        self.assertMapped(
            pi_event("session_compact", reason="manual", idle=True), "PostCompact", sidepulse_status="ready"
        )
        self.assertIsNone(normalize_payload(pi_event("session_compact_failed", aborted=False, idle=False)))
        self.assertMapped(
            pi_event("session_compact_failed", aborted=True, idle=True), "PostCompact", sidepulse_status="ready"
        )
        self.assertMapped(
            pi_event("session_compact_failed", aborted=False, error_message="overflow", idle=True),
            "PostCompact",
            sidepulse_status="error",
            message="overflow",
        )

    def test_settled_run_outcomes(self) -> None:
        self.assertMapped(
            pi_event("agent_settled", aborted=False, stop_reason="stop", assistant_text="Done."),
            "Stop",
            last_assistant_message="Done.",
        )
        self.assertMapped(
            pi_event("agent_settled", aborted=False, stop_reason="error", error_message="429 rate limited"),
            "StopFailure",
            error_details="429 rate limited",
        )
        self.assertMapped(pi_event("agent_settled", aborted=True), "Interrupt")
        self.assertMapped(pi_event("agent_settled", aborted=False, stop_reason="aborted"), "Interrupt")
        self.assertMapped(pi_event("agent_settled", aborted=False), "Stop", last_assistant_message="")

    def test_unknown_or_missing_events_are_ignored(self) -> None:
        self.assertIsNone(normalize_payload(pi_event("agent_end")))
        self.assertIsNone(normalize_payload({"session_id": "x"}))
        self.assertIsNone(normalize_payload({"pi_event": 42}))

    def test_events_drive_sidepulse_states(self) -> None:
        cases = [
            (pi_event("session_start", reason="startup", idle=True), AgentMode.IDLE_READY),
            (pi_event("before_agent_start", message_type="user", prompt="fix the test"), AgentMode.WORKING),
            (pi_event("tool_execution_start", tool_name="bash", args={"command": "pytest"}), AgentMode.TOOL_RUNNING),
            (pi_event("ui_prompt_start", kind="confirm", title="Allow?"), AgentMode.WAITING_FOR_INPUT),
            (pi_event("ui_prompt_end"), AgentMode.WORKING),
            (pi_event("tool_execution_end", tool_name="bash", is_error=True), AgentMode.BLOCKED_ERROR),
            (pi_event("tool_execution_end", tool_name="read", is_error=False), AgentMode.WORKING),
            (
                pi_event("agent_settled", stop_reason="stop", assistant_text="Should I add a LICENSE?", idle=True),
                AgentMode.WAITING_FOR_INPUT,
            ),
            (pi_event("before_agent_start", message_type="user", prompt="no"), AgentMode.WORKING),
            (pi_event("agent_settled", stop_reason="stop", assistant_text="All done.", idle=True), AgentMode.COMPLETED),
            (pi_event("session_before_compact", reason="manual", idle=True), AgentMode.WORKING),
            (pi_event("session_compact", reason="manual", idle=True), AgentMode.IDLE_READY),
            (pi_event("agent_settled", stop_reason="error", error_message="boom", idle=True), AgentMode.BLOCKED_ERROR),
            (pi_event("agent_settled", aborted=True, idle=True), AgentMode.IDLE_READY),
            (pi_event("session_shutdown", reason="quit", idle=True), AgentMode.COMPLETED),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "pi.jsonl"
            for payload, expected in cases:
                row = normalize_payload(payload)
                row["logged_at"] = datetime.now(timezone.utc).isoformat()
                row["agent_origin"] = "pi CLI"
                with log.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                snapshot = AgentMonitor(
                    sources=(SourceSpec("pi", log),),
                    stale_after_seconds=3600,
                ).snapshot(include_stale=True)
                # Idle and completed sessions age into stale_statuses; the mode is what matters.
                [status] = snapshot.statuses + snapshot.stale_statuses
                with self.subTest(event=payload["pi_event"]):
                    self.assertEqual(status.mode, expected)

            self.assertEqual(status.provider, "pi")
            self.assertEqual(status.session_id, "pi-session-1234")
            self.assertEqual(status.display_name, "project: fix the test (pi-sessi)")
            self.assertEqual(status.origin, "pi CLI")


class PiHookEntryTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.log = self.home / "pi.jsonl"
        env = patch.dict(
            os.environ,
            {
                "HOME": str(self.home),
                "XDG_STATE_HOME": str(self.home / "state"),
                "SIDEPULSE_DISABLE_EVENT_SOCKET": "1",
                "SIDEPULSE_AGENT_ORIGIN": "pi CLI",
            },
        )
        env.start()
        self.addCleanup(env.stop)

    def run_hook(self, payload: str) -> int:
        with patch.object(sys, "stdin", io.StringIO(payload)), redirect_stdout(io.StringIO()):
            return hook_log_main("pi", self.log)

    def rows(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_pi_events_are_normalized_before_logging(self) -> None:
        code = self.run_hook(json.dumps(pi_event("tool_execution_start", tool_name="bash", args={"command": "ls"})))

        self.assertEqual(code, 0)
        [row] = self.rows()
        self.assertEqual(row["hook_event_name"], "PreToolUse")
        self.assertEqual(row["tool_input"], {"command": "ls"})
        self.assertEqual(row["agent_origin"], "pi CLI")
        self.assertIn("logged_at", row)

    def test_events_that_change_nothing_write_nothing(self) -> None:
        code = self.run_hook(json.dumps(pi_event("session_compact_failed", aborted=False, idle=False)))

        self.assertEqual(code, 0)
        self.assertEqual(self.rows(), [])

    def test_malformed_payload_still_leaves_a_breadcrumb(self) -> None:
        self.assertEqual(self.run_hook("not json"), 0)
        [row] = self.rows()
        self.assertEqual(row["hook_event_name"], "ParseError")


class PiSessionTests(unittest.TestCase):
    def test_label_keeps_pi_lowercase(self) -> None:
        self.assertEqual(provider_label("pi"), "pi")

    def test_session_can_resume_in_pi(self) -> None:
        status = AgentStatus(
            provider="pi",
            agent_id="pi:session:01a1c762",
            display_name="pi session",
            mode=AgentMode.COMPLETED,
            updated_at=datetime.now(timezone.utc),
            event_name="Stop",
            session_id="01a1c762",
            cwd="/tmp/project path",
        )

        self.assertEqual(session_resume_command(status), "cd '/tmp/project path' && pi --session 01a1c762")

    def test_pi_cli_process_is_recognised(self) -> None:
        pi_cli = ProcessInfo(10, 9, "node", "node /opt/homebrew/bin/pi --model x")
        bundled = ProcessInfo(10, 9, "/opt/homebrew/bin/node", "/opt/homebrew/bin/node /usr/lib/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js")
        other_node = ProcessInfo(10, 9, "node", "node server.js")
        shell = ProcessInfo(9, 1, "-zsh", "-zsh")

        self.assertTrue(is_pi_process(pi_cli))
        self.assertTrue(is_pi_process(bundled))
        self.assertFalse(is_pi_process(other_node))
        self.assertEqual(origin_from_processes("pi", (pi_cli, shell)).label, "pi CLI")
        self.assertIsNone(origin_from_processes("pi", (other_node, shell)))


def node_strips_typescript() -> bool:
    node = shutil.which("node")
    if node is None:
        return False
    try:
        version = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10).stdout
        return int(version.strip().lstrip("v").split(".")[0]) >= 23
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


@unittest.skipUnless(node_strips_typescript(), "needs Node.js 23+ to run the TypeScript extension")
class PiExtensionRuntimeTests(unittest.TestCase):
    """Run the generated extension under Node with a fake pi and a recording hook."""

    HARNESS = textwrap.dedent(
        """
        const [extensionPath] = process.argv.slice(2);
        const { default: register } = await import(extensionPath);
        const handlers = {};
        register({ on: (name, fn) => { (handlers[name] ??= []).push(fn); } });
        let idle = true;
        const ctx = {
          cwd: "/tmp/project", mode: "tui", model: { id: "test-model" },
          sessionManager: { getSessionId: () => "pi-session-1234", getSessionFile: () => undefined },
          isIdle: () => idle,
        };
        const emit = async (name, event = {}) => {
          for (const handler of handlers[name] ?? []) await handler({ type: name, ...event }, ctx);
        };
        await emit("session_start", { reason: "startup" });
        idle = false;
        await emit("before_agent_start", { messageType: "user", prompt: "fix the test" });
        for (let i = 0; i < 20; i++) {
          await emit("tool_execution_start", { toolName: "read", toolCallId: `t${i}`, args: { path: `f${i}`, big: { nested: true } } });
          await emit("tool_execution_end", { toolName: "read", toolCallId: `t${i}`, isError: false, result: {} });
        }
        await emit("agent_end", { messages: [
          { role: "user", content: "fix the test" },
          { role: "assistant", stopReason: "stop", content: [{ type: "thinking", thinking: "hm" }, { type: "text", text: "Fixed it." }] },
        ] });
        idle = true;
        await emit("agent_settled", { aborted: false });
        await emit("session_shutdown", { reason: "quit" });
        """
    )
    RECORDER = "import sys\nwith open(sys.argv[1], 'a') as f:\n    f.write(sys.stdin.read().replace('\\n', ' ') + '\\n')\n"

    def test_extension_forwards_events_in_order_and_collapses_tool_bursts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = root / "recorder.py"
            recorder.write_text(self.RECORDER)
            out = root / "events.jsonl"
            extension = root / "sidepulse.ts"
            extension.write_text(render_pi_extension([sys.executable, str(recorder), str(out)]))
            harness = root / "harness.mjs"
            harness.write_text(self.HARNESS)

            env = {key: value for key, value in os.environ.items() if key != "PI_SIDEPULSE"}
            result = subprocess.run(
                [shutil.which("node"), "--no-warnings", str(harness), str(extension)],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            events = [json.loads(line) for line in out.read_text().splitlines()]
            names = [event["pi_event"] for event in events]
            lifecycle = [name for name in names if not name.startswith("tool_")]
            self.assertEqual(
                lifecycle,
                ["session_start", "before_agent_start", "agent_settled", "session_shutdown"],
            )
            self.assertLess(len(names), 4 + 40, "queued tool events were not collapsed")

            settled = events[names.index("agent_settled")]
            self.assertEqual(settled["assistant_text"], "Fixed it.")
            self.assertEqual(settled["stop_reason"], "stop")
            self.assertTrue(settled["idle"])
            self.assertEqual(events[0]["session_id"], "pi-session-1234")
            self.assertEqual(events[0]["model"], "test-model")
            for event in events:
                if event["pi_event"] == "tool_execution_start":
                    self.assertEqual(set(event["args"]), {"path"})  # nested values dropped

    def test_pi_sidepulse_zero_disables_the_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = root / "recorder.py"
            recorder.write_text(self.RECORDER)
            out = root / "events.jsonl"
            extension = root / "sidepulse.ts"
            extension.write_text(render_pi_extension([sys.executable, str(recorder), str(out)]))
            harness = root / "harness.mjs"
            harness.write_text(self.HARNESS)

            result = subprocess.run(
                [shutil.which("node"), "--no-warnings", str(harness), str(extension)],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "PI_SIDEPULSE": "0"},
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
