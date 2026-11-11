"""Emergency alerts: store, CLI, IPC, background service, and menu-bar app."""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sidepulse import alerts
from sidepulse.alerts import (
    ALERT_PROGRAM,
    DEFAULT_ALERT_MESSAGE,
    alert_program,
    dismiss_alert,
    dismiss_all_alerts,
    load_alerts,
    message_from_payload,
    raise_alert,
)
from sidepulse.cli import sidepulse_main
from sidepulse.ipc import HookEventServer, notify_alerts_changed
from sidepulse.models import AgentMode
from sidepulse.settings import AgentMonitorSettings


class AlertStoreTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "alerts.json"
        self.paths = (self.path,)

    def raise_(self, message, **kwargs):
        return raise_alert(message, path=self.path, notify=False, **kwargs)


class AlertStoreTests(AlertStoreTestCase):
    def test_raise_list_and_dismiss(self):
        first = self.raise_("Disk almost full", source="backup")
        second = self.raise_("Build failed", source="ci")

        self.assertEqual(
            [alert.alert_id for alert in load_alerts(self.paths)],
            [first.alert_id, second.alert_id],
        )
        self.assertTrue(dismiss_alert(first.alert_id, paths=self.paths, notify=False))
        self.assertFalse(dismiss_alert(first.alert_id, paths=self.paths, notify=False))
        self.assertEqual(load_alerts(self.paths), [second])
        self.assertEqual(dismiss_all_alerts(paths=self.paths, notify=False), 1)
        self.assertEqual(load_alerts(self.paths), [])

    def test_alerts_persist_until_dismissed(self):
        alert = self.raise_("Disk almost full")
        document = json.loads(self.path.read_text())
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["alerts"][0]["id"], alert.alert_id)
        # A later process (the app after a restart) still sees it.
        self.assertEqual(load_alerts(self.paths), [alert])

    def test_repeats_update_one_alert(self):
        now = datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc)
        first = self.raise_("Disk almost full", source="backup", now=now)
        again = self.raise_(
            "Disk almost full",
            source="backup",
            now=now.replace(minute=5),
        )
        other_source = self.raise_("Disk almost full", source="monitor")

        self.assertEqual(again.alert_id, first.alert_id)
        self.assertEqual(again.count, 2)
        self.assertEqual(again.raised_at, "2026-11-11T09:00:00Z")
        self.assertEqual(again.last_raised_at, "2026-11-11T09:05:00Z")
        self.assertNotEqual(other_source.alert_id, first.alert_id)
        self.assertEqual(len(load_alerts(self.paths)), 2)

    def test_explicit_key_replaces_the_message(self):
        first = self.raise_("Disk 90% full", key="disk")
        second = self.raise_("Disk 97% full", key="disk")

        self.assertEqual(second.alert_id, first.alert_id)
        self.assertEqual([alert.message for alert in load_alerts(self.paths)], ["Disk 97% full"])

    def test_messages_are_one_line_and_bounded(self):
        alert = self.raise_("  first line\n\n  second\tline  ")
        self.assertEqual(alert.message, "first line second line")
        self.assertEqual(self.raise_("   ").message, DEFAULT_ALERT_MESSAGE)
        long_alert = self.raise_("x" * 2000)
        self.assertEqual(len(long_alert.message), alerts.MAX_MESSAGE_CHARS)
        self.assertTrue(long_alert.message.endswith("…"))

    def test_unique_alerts_are_capped_keeping_the_newest(self):
        with patch.object(alerts, "MAX_ACTIVE_ALERTS", 3):
            for index in range(5):
                self.raise_(f"alert {index}")
        self.assertEqual(
            [alert.message for alert in load_alerts(self.paths)],
            ["alert 2", "alert 3", "alert 4"],
        )

    def test_unreadable_store_reads_as_empty_and_is_replaced(self):
        self.path.write_text("{not json")
        self.assertEqual(load_alerts(self.paths), [])
        alert = self.raise_("Recovered")
        self.assertEqual(load_alerts(self.paths), [alert])

    def test_reads_and_dismisses_across_state_dirs(self):
        # The CLI and the LaunchAgent app can resolve different state dirs.
        other = self.root / "other" / "alerts.json"
        here = self.raise_("From the shell")
        there = raise_alert("From the app", path=other, notify=False)
        both = (self.path, other)

        self.assertEqual({alert.alert_id for alert in load_alerts(both)}, {here.alert_id, there.alert_id})
        self.assertTrue(dismiss_alert(there.alert_id, paths=both, notify=False))
        self.assertEqual(load_alerts(both), [here])

    def test_raising_and_dismissing_notify_the_app(self):
        with patch.object(alerts, "notify_alerts_changed") as notify:
            alert = raise_alert("Ping", path=self.path)
            notify.assert_called_once_with()
            dismiss_alert(alert.alert_id, paths=self.paths)
            self.assertEqual(notify.call_count, 2)
            dismiss_all_alerts(paths=self.paths)  # Nothing left: no notification.
            self.assertEqual(notify.call_count, 2)

    def test_alert_program_is_a_red_pulse_at_device_brightness(self):
        self.assertIn("#FF0000", ALERT_PROGRAM)
        self.assertTrue(ALERT_PROGRAM.endswith("repeat"))
        self.assertEqual(alert_program(), ALERT_PROGRAM)
        self.assertTrue(alert_program(128).startswith("brightness 128\n"))

    def test_message_from_hook_payload_or_text(self):
        self.assertEqual(
            message_from_payload(
                '{"hook_event_name": "Notification", "message": "Claude needs permission"}'
            ),
            ("Claude needs permission", "Notification"),
        )
        self.assertEqual(
            message_from_payload('{"hook_event_name": "StopFailure"}'),
            ("StopFailure", "StopFailure"),
        )
        self.assertEqual(message_from_payload("  plain\ntext \n"), ("plain text", ""))
        self.assertEqual(message_from_payload("{broken"), ("{broken", ""))


class AlertCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        environment = {key: value for key, value in os.environ.items() if key != "XDG_STATE_HOME"}
        environment["HOME"] = directory.name
        for context in (
            patch.dict(os.environ, environment, clear=True),
            patch.object(alerts, "notify_alerts_changed"),
        ):
            context.start()
            self.addCleanup(context.stop)

    def run_cli(self, *args, stdin: str | None = None):
        out, err = io.StringIO(), io.StringIO()
        fake_stdin = io.StringIO(stdin or "")
        fake_stdin.isatty = lambda: stdin is None
        with redirect_stdout(out), redirect_stderr(err), patch("sys.stdin", fake_stdin):
            code = sidepulse_main(["alert", *args])
        return code, out.getvalue(), err.getvalue()

    def test_raise_list_dismiss(self):
        code, out, _ = self.run_cli("raise", "Disk", "almost", "full", "--source", "backup")
        self.assertEqual(code, 0)
        self.assertIn("Disk almost full", out)
        _, out, _ = self.run_cli("raise", "Disk almost full", "--source", "backup")
        self.assertIn("raised 2 times", out)

        code, out, _ = self.run_cli("list", "--json")
        listed = json.loads(out)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["source"], "backup")
        _, out, _ = self.run_cli("list")
        self.assertIn("backup, 2 times", out)

        code, _, err = self.run_cli("dismiss", "missing")
        self.assertEqual(code, 1)
        self.assertIn("No active alert", err)
        code, out, _ = self.run_cli("dismiss", listed[0]["id"])
        self.assertEqual(code, 0)
        _, out, _ = self.run_cli("list")
        self.assertEqual(out.strip(), "No active alerts.")

    def test_raise_reads_an_agent_hook_payload_from_stdin_quietly(self):
        code, out, _ = self.run_cli(
            "raise",
            "--quiet",
            stdin='{"hook_event_name": "Notification", "message": "Approve rm -rf?"}',
        )
        self.assertEqual((code, out), (0, ""))
        listed = json.loads(self.run_cli("list", "--json")[1])
        self.assertEqual(
            (listed[0]["message"], listed[0]["source"]),
            ("Approve rm -rf?", "Notification"),
        )

    def test_dismiss_needs_ids_or_all(self):
        self.run_cli("raise", "one")
        self.run_cli("raise", "two")
        self.assertEqual(self.run_cli("dismiss")[0], 2)
        self.assertEqual(self.run_cli("dismiss", "x", "--all")[0], 2)
        code, out, _ = self.run_cli("dismiss", "--all")
        self.assertEqual((code, out.strip()), (0, "Dismissed 2 alerts."))


class AlertIpcTests(unittest.TestCase):
    def test_app_socket_handles_alerts_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            changed = []
            events = []
            server = HookEventServer(
                lambda provider, line: events.append((provider, line)),
                socket_path=Path(tmp) / "events.sock",
                on_alerts_changed=lambda: changed.append(True),
            )
            server.start()
            try:
                self.assertTrue(notify_alerts_changed(socket_path=server.socket_path, timeout=0.5))
                deadline = time.time() + 1
                while not changed and time.time() < deadline:
                    time.sleep(0.01)
            finally:
                server.stop()
            self.assertEqual(changed, [True])
            self.assertEqual(events, [])

    def test_notify_without_a_running_app_is_harmless(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(notify_alerts_changed(socket_path=Path(tmp) / "none.sock"))


class AlertServiceTests(unittest.TestCase):
    def sync(self, active_alerts):
        from sidepulse import service

        written = []
        candidate = SimpleNamespace(
            root=Path("/Volumes/SidePulsePro"),
            target=Path("/Volumes/SidePulsePro/LEDS.LED"),
        )
        link = SimpleNamespace(link_id="phone1")
        settings = AgentMonitorSettings()
        sent = []
        with (
            patch.object(service, "status_bar_monitor_available", return_value=False),
            patch.object(service, "load_settings", return_value=settings),
            patch.object(service, "load_alerts", return_value=active_alerts),
            patch.object(service, "read_battery_snapshot", side_effect=OSError),
            patch.object(service, "discover_devices", return_value=[candidate]),
            patch.object(
                service,
                "write_led_program",
                side_effect=lambda program, device_path: written.append(program),
            ),
            patch.object(service, "load_ios_links", return_value=[link]),
            patch.object(
                service,
                "send_ios_program",
                side_effect=lambda link, program, data: sent.append(program),
            ),
            patch.object(service, "HookEventServer"),
        ):
            service.SidePulseService().sync_outputs()
        return written, sent

    def test_headless_service_pulses_red_while_alerting(self):
        written, sent = self.sync([SimpleNamespace(alert_id="a1")])
        self.assertEqual(written, [ALERT_PROGRAM])
        self.assertEqual(sent, [ALERT_PROGRAM])

        written, sent = self.sync([])
        self.assertNotIn(ALERT_PROGRAM, written)
        self.assertNotIn(ALERT_PROGRAM, sent)


try:
    from sidepulse import status_bar as sb
except (ImportError, SystemExit) as exc:  # AppKit and PyObjC are macOS-only.
    sb = None
    STATUS_BAR_SKIP = str(exc)


class AlertStatusBarTests(AlertStoreTestCase):
    @classmethod
    def setUpClass(cls):
        if sb is None:
            raise unittest.SkipTest(STATUS_BAR_SKIP)
        from AppKit import NSApplication

        NSApplication.sharedApplication()

    def device(self, root: Path, *, device_id: str | None = None, display: str = "agent"):
        return sb.StatusBarDevice(
            device_id=device_id or str(root),
            name="SidePulse Pro",
            root=root,
            target=root / "LEDS.LED",
            connected=True,
            display=display,
            brightness=255,
        )

    def controller(self):
        controller = sb.StatusBarController.alloc().init()
        controller.settings = AgentMonitorSettings(virtual_status_device_enabled=True)
        controller.leds_enabled = True
        return controller

    def test_menu_lists_alerts_with_dismiss_actions(self):
        first = self.raise_("Disk almost full", source="backup")
        second = self.raise_("Build failed")
        target = SimpleNamespace(
            settings=AgentMonitorSettings(),
            closed_lid_awake=SimpleNamespace(last_error=None),
            status_bar_devices=lambda: [],
            active_alerts=[first, second],
        )
        snapshot = SimpleNamespace(statuses=[], collected_at=datetime.now(timezone.utc))
        menu = sb.build_menu(snapshot, sb.STATE_IDLE, target)
        titles = [str(menu.itemAtIndex_(index).title()) for index in range(menu.numberOfItems())]

        self.assertEqual(titles[0], "SidePulse")
        self.assertTrue(titles[2].startswith("⚠︎ ALERT: Disk almost full"))
        self.assertTrue(titles[3].startswith("⚠︎ ALERT: Build failed"))
        self.assertEqual(titles[4], "Dismiss All Alerts")
        self.assertLess(titles.index("Dismiss All Alerts"), titles.index("Agents"))
        self.assertEqual(menu.itemAtIndex_(4).action(), "dismissAllAlerts:")

        row = menu.itemAtIndex_(2)
        color = row.attributedTitle().attribute_atIndex_effectiveRange_(
            sb.NSForegroundColorAttributeName, 0, None
        )[0]
        self.assertEqual(color, sb.NSColor.systemRedColor())
        submenu = row.submenu()
        sub_titles = [str(submenu.itemAtIndex_(i).title()) for i in range(submenu.numberOfItems())]
        self.assertIn("From backup", sub_titles)
        dismiss = submenu.itemWithTitle_("Dismiss")
        self.assertEqual(dismiss.action(), "dismissAlert:")
        self.assertEqual(dismiss.representedObject(), first.alert_id)
        self.assertIs(dismiss.target(), target)

        target.active_alerts = [first]
        titles = [
            str(item.title())
            for item in sb.build_menu(snapshot, sb.STATE_IDLE, target).itemArray()
        ]
        self.assertNotIn("Dismiss All Alerts", titles)
        target.active_alerts = []
        titles = [
            str(item.title())
            for item in sb.build_menu(snapshot, sb.STATE_IDLE, target).itemArray()
        ]
        self.assertFalse(any("ALERT" in title for title in titles))

    def test_physical_device_pulses_red_then_returns_to_its_state(self):
        root = self.root / "SidePulsePro"
        root.mkdir()
        controller = self.controller()
        device = self.device(root)
        with patch.object(sb.StatusBarController, "status_bar_devices", return_value=[device]):
            controller.update_active_alerts([self.raise_("Disk almost full")])
            controller.sync_leds_now(AgentMode.WORKING, None, sb.LED_DISPLAY_AGENT)
            self.assertEqual(device.target.read_text(), ALERT_PROGRAM)

            controller.update_active_alerts([])
            controller.sync_leds_now(AgentMode.WORKING, None, sb.LED_DISPLAY_AGENT)
            self.assertNotEqual(device.target.read_text(), ALERT_PROGRAM)

    def test_custom_display_devices_are_left_alone(self):
        root = self.root / "SidePulsePro"
        root.mkdir()
        controller = self.controller()
        device = self.device(root, display=sb.LED_DISPLAY_CUSTOM)
        with patch.object(sb.StatusBarController, "status_bar_devices", return_value=[device]):
            controller.update_active_alerts([self.raise_("Disk almost full")])
            controller.sync_leds_now(AgentMode.WORKING, None, sb.LED_DISPLAY_AGENT)
        self.assertFalse(device.target.exists())

    def test_linked_phone_gets_the_alert(self):
        controller = self.controller()
        phone = sb.StatusBarDevice(
            device_id="ios/phone1",
            name="iPhone",
            root=Path("ios/phone1"),
            target=Path("ios/phone1"),
            connected=True,
            display="agent",
            brightness=255,
            remote_link=SimpleNamespace(link_id="phone1"),
        )
        with (
            patch.object(sb.StatusBarController, "status_bar_devices", return_value=[phone]),
            patch.object(sb, "send_ios_program") as send,
        ):
            controller.update_active_alerts([self.raise_("Disk almost full")])
            controller.sync_leds_now(AgentMode.IDLE_READY, None, sb.LED_DISPLAY_AGENT)
        self.assertEqual(send.call_args.args[1], ALERT_PROGRAM)

    def test_notch_pulses_red_without_rainbow(self):
        controller = self.controller()
        controller.settings = controller.settings.with_rainbow_theme(True)
        controller.virtual_status_device = Mock()
        notch = self.device(Path(sb.VIRTUAL_DEVICE_ID), device_id=sb.VIRTUAL_DEVICE_ID)
        controller.active_alerts = [self.raise_("Disk almost full")]
        with patch.object(sb.StatusBarController, "status_bar_devices", return_value=[notch]):
            controller.sync_virtual_status_device(AgentMode.WORKING, None)
        controller.virtual_status_device.set_rainbow.assert_called_with(False)
        controller.virtual_status_device.set_program.assert_called_with(ALERT_PROGRAM)

    def test_alerts_override_lid_hold_and_running_animations(self):
        controller = self.controller()
        controller.lid_closed_led_hold_active = True
        controller.led_animation_until_monotonic = time.monotonic() + 60
        controller.virtual_status_device = Mock()
        with (
            patch.object(sb.StatusBarController, "should_hold_lid_closed_led_state", return_value=True),
            patch("sidepulse.status_bar.threading.Thread") as thread,
        ):
            controller.sync_leds(AgentMode.IDLE_READY, None, sb.LED_DISPLAY_AGENT)
            thread.assert_not_called()

            controller.update_active_alerts([self.raise_("Disk almost full")])
            self.assertEqual(controller.led_animation_until_monotonic, 0.0)
            self.assertTrue(controller.led_sync_retry_pending)
            controller.sync_leds(AgentMode.IDLE_READY, None, sb.LED_DISPLAY_AGENT)
            thread.assert_called_once()
            self.assertFalse(controller.led_sync_retry_pending)
        self.assertTrue(controller.lid_closed_led_hold_active)

    def test_dismissing_while_the_lid_is_closed_fades_back_to_dark(self):
        controller = self.controller()
        controller.lid_closed_led_hold_active = True
        controller.active_alerts = [self.raise_("Disk almost full")]
        with (
            patch.object(sb.StatusBarController, "should_hold_lid_closed_led_state", return_value=True),
            patch.object(sb.StatusBarController, "play_lid_animation") as play,
        ):
            controller.update_active_alerts([])
        play.assert_called_once_with(sb.LID_ANIMATION_CLOSED)

    def test_lid_animations_and_previews_wait_for_dismissal(self):
        controller = self.controller()
        controller.active_alerts = [self.raise_("Disk almost full")]
        with patch("sidepulse.status_bar.threading.Thread") as thread:
            controller.play_lid_animation(sb.LID_ANIMATION_OPEN)
            controller.show_agent_animation_on_device(AgentMode.WORKING.value)
        thread.assert_not_called()

    def test_sync_retries_when_alerts_change_mid_sync(self):
        controller = self.controller()
        controller.led_sync_retry_pending = True
        with (
            patch.object(sb.StatusBarController, "sync_leds_now"),
            patch.object(sb.StatusBarController, "schedule_event_refresh") as refresh,
        ):
            controller.sync_leds_worker(AgentMode.WORKING, None, sb.LED_DISPLAY_AGENT)
        refresh.assert_called_once_with()
        self.assertFalse(controller.led_sync_in_flight)
        self.assertFalse(controller.led_sync_retry_pending)

    def test_menu_actions_dismiss_and_refresh(self):
        controller = self.controller()
        sender = SimpleNamespace(representedObject=lambda: "abc123")
        with (
            patch.object(sb, "dismiss_alert") as dismiss,
            patch.object(sb, "dismiss_all_alerts") as dismiss_all,
            patch.object(sb.StatusBarController, "refresh_") as refresh,
        ):
            controller.dismissAlert_(sender)
            dismiss.assert_called_once_with("abc123", notify=False)
            controller.dismissAllAlerts_(None)
            dismiss_all.assert_called_once_with(notify=False)
        self.assertEqual(refresh.call_count, 2)

    def test_status_icon_turns_into_a_red_warning(self):
        controller = self.controller()
        self.addCleanup(controller.stop_status_icon_animation)
        controller.status_item = Mock()
        button = controller.status_item.button.return_value
        controller.active_alerts = [self.raise_("Disk almost full")]
        controller.set_status(sb.STATE_IDLE)

        button.setToolTip_.assert_called_with("SidePulse ALERT: Disk almost full")
        icon = sb.alert_status_icon()
        self.assertIsNotNone(icon)
        self.assertFalse(icon.isTemplate())
        self.assertIsNot(sb.alert_status_icon(0), sb.alert_status_icon(24))


if __name__ == "__main__":
    unittest.main()
