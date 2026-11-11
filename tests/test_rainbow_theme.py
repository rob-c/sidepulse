"""Rainbow theme: the Rainbow animations on every SidePulse device and the Notch."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sidepulse.led_status import program_for_agent_mode
from sidepulse.models import AgentMode
from sidepulse.settings import (
    AGENT_ANIMATION_PROFILE_PURPLE,
    AGENT_ANIMATION_PROFILE_RAINBOW,
    ANIMATION_STATES,
    AgentMonitorSettings,
    builtin_agent_animation_profiles,
    load_settings,
    save_settings,
)

RAINBOW = builtin_agent_animation_profiles()[AGENT_ANIMATION_PROFILE_RAINBOW].animations


def purple_settings(**kwargs) -> AgentMonitorSettings:
    return AgentMonitorSettings(**kwargs).with_applied_agent_animation_profile(
        AGENT_ANIMATION_PROFILE_PURPLE
    )


def working_program(style: str, led_count: int) -> str:
    return program_for_agent_mode(
        AgentMode.WORKING, led_count=led_count, animation_style=style
    ).strip()


class RainbowThemeSettingTests(unittest.TestCase):
    def test_turning_it_on_uses_rainbow_for_every_state_and_off_restores(self):
        settings = purple_settings()
        self.assertFalse(settings.rainbow_theme_enabled)

        rainbow = settings.with_rainbow_theme(True)
        self.assertTrue(rainbow.rainbow_theme_enabled)
        self.assertEqual(
            {mode: rainbow.agent_animation_id(mode) for mode in ANIMATION_STATES}, RAINBOW
        )
        self.assertEqual(rainbow.matching_agent_animation_profile_id(), AGENT_ANIMATION_PROFILE_RAINBOW)
        self.assertIs(rainbow.with_rainbow_theme(True), rainbow)

        restored = rainbow.with_rainbow_theme(False)
        self.assertFalse(restored.rainbow_theme_enabled)
        self.assertEqual(restored.matching_agent_animation_profile_id(), AGENT_ANIMATION_PROFILE_PURPLE)
        self.assertEqual(restored.rainbow_restore_animations, {})

    def test_choosing_the_rainbow_profile_ticks_the_theme(self):
        settings = AgentMonitorSettings().with_applied_agent_animation_profile(
            AGENT_ANIMATION_PROFILE_RAINBOW
        )
        self.assertTrue(settings.rainbow_theme_enabled)
        # Nothing was remembered, so turning it off falls back to the default profile.
        self.assertEqual(settings.with_rainbow_theme(False).agent_animation_id("working"), "cyan-roll")

    def test_restore_point_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            save_settings(purple_settings().with_rainbow_theme(True), path)
            document = json.loads(path.read_text())
            self.assertNotIn("rainbow_theme", document)
            self.assertEqual(document["rainbow_restore"]["working"]["style"], "purple-tide")

            loaded = load_settings(path)
            self.assertTrue(loaded.rainbow_theme_enabled)
            self.assertEqual(
                loaded.with_rainbow_theme(False).matching_agent_animation_profile_id(),
                AGENT_ANIMATION_PROFILE_PURPLE,
            )

    def test_earlier_notch_only_setting_turns_into_the_full_theme(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            save_settings(purple_settings(), path)
            document = json.loads(path.read_text())
            document["rainbow_theme"] = True
            path.write_text(json.dumps(document))

            loaded = load_settings(path)
            self.assertTrue(loaded.rainbow_theme_enabled)
            save_settings(loaded.with_rainbow_theme(False), path)
            reloaded = load_settings(path)
            self.assertFalse(reloaded.rainbow_theme_enabled)
            self.assertEqual(
                reloaded.matching_agent_animation_profile_id(), AGENT_ANIMATION_PROFILE_PURPLE
            )


class RainbowServiceTests(unittest.TestCase):
    def test_headless_service_writes_rainbow_to_devices(self):
        from sidepulse import service

        written = {}
        pro = SimpleNamespace(root=Path("/Volumes/SidePulse"), target=Path("/Volumes/SidePulse/LEDS.LED"))
        dot = SimpleNamespace(root=Path("/Volumes/PulseDot"), target=Path("/Volumes/PulseDot/LEDS.LED"))
        with (
            patch.object(service, "status_bar_monitor_available", return_value=False),
            patch.object(service, "load_settings", return_value=purple_settings().with_rainbow_theme(True)),
            patch.object(service, "load_alerts", return_value=[]),
            patch.object(service, "read_battery_snapshot", side_effect=OSError),
            patch.object(service, "discover_devices", return_value=[pro, dot]),
            patch.object(
                service,
                "write_led_program",
                side_effect=lambda program, device_path: written.__setitem__(device_path, program),
            ),
            patch.object(service, "load_ios_links", return_value=[]),
            patch.object(service, "HookEventServer"),
        ):
            instance = service.SidePulseService()
            instance.monitor = Mock()
            instance.monitor.snapshot.return_value = SimpleNamespace(
                aggregate=SimpleNamespace(mode=AgentMode.WORKING)
            )
            instance.sync_outputs()
        self.assertEqual(written[pro.target].strip(), working_program("rainbow-spin", 8))
        self.assertEqual(written[dot.target].strip(), working_program("rainbow-spin", 2))


try:
    from sidepulse import status_bar as sb
except (ImportError, SystemExit) as exc:  # AppKit and PyObjC are macOS-only.
    sb = None
    STATUS_BAR_SKIP = str(exc)


class RainbowStatusBarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sb is None:
            raise unittest.SkipTest(STATUS_BAR_SKIP)
        from AppKit import NSApplication

        NSApplication.sharedApplication()

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def device(self, folder: str, device_id: str | None = None):
        root = self.root / folder
        root.mkdir(exist_ok=True)
        return sb.StatusBarDevice(
            device_id=device_id or str(root),
            name=folder,
            root=root,
            target=root / "LEDS.LED",
            connected=True,
            display="agent",
            brightness=255,
        )

    def test_theme_reaches_sidepulse_dot_and_notch_then_restores(self):
        pro = self.device("SidePulse")
        dot = self.device("PulseDot")
        notch = self.device("notch", device_id=sb.VIRTUAL_DEVICE_ID)
        controller = sb.StatusBarController.alloc().init()
        controller.settings = purple_settings(virtual_status_device_enabled=True)
        controller.leds_enabled = True
        controller.virtual_status_device = Mock()
        controller.settings_window = sb.build_settings_window(controller)
        self.addCleanup(controller.settings_window.close)
        checkbox = controller.settings_buttons["rainbow_theme"]

        def sync():
            controller.sync_leds_now(AgentMode.WORKING, None, sb.LED_DISPLAY_AGENT)
            controller.sync_virtual_status_device(AgentMode.WORKING, None)

        with (
            patch.object(sb.StatusBarController, "status_bar_devices", return_value=[pro, dot, notch]),
            patch.object(sb, "save_settings") as save,
            patch.object(sb.StatusBarController, "refresh_"),
        ):
            sync()
            self.assertEqual(pro.target.read_text().strip(), working_program("purple-tide", 8))

            checkbox.setState_(sb.NSOnState)
            controller.setRainbowThemeFromCheckbox_(checkbox)
            self.assertTrue(save.call_args.args[0].rainbow_theme_enabled)
            self.assertEqual(controller.selected_agent_animation_profile_id, AGENT_ANIMATION_PROFILE_RAINBOW)
            profile_popup = controller.settings_fields["agent_animation_profile"]
            self.assertEqual(str(profile_popup.titleOfSelectedItem()), "Rainbow")
            sync()
            self.assertEqual(pro.target.read_text().strip(), working_program("rainbow-spin", 8))
            self.assertEqual(dot.target.read_text().strip(), working_program("rainbow-spin", 2))
            notch_program = controller.virtual_status_device.set_program.call_args.args[0]
            self.assertEqual(notch_program.strip(), working_program("rainbow-spin", 8))
            # The previews show the same program the devices get.
            preview = controller.settings_fields[f"agent_animation_preview_{AgentMode.WORKING.value}"]
            self.assertIn("roll-right", preview.current_program)

            controller.toggleRainbowTheme_(None)
            self.assertEqual(checkbox.state(), sb.NSOffState)
            sync()
            self.assertEqual(pro.target.read_text().strip(), working_program("purple-tide", 8))
            self.assertEqual(
                controller.virtual_status_device.set_program.call_args.args[0].strip(),
                working_program("purple-tide", 8),
            )

    def test_menu_toggle_lives_in_the_main_menu(self):
        notch = sb.StatusBarDevice(
            device_id=sb.VIRTUAL_DEVICE_ID,
            name=sb.VIRTUAL_DEVICE_NAME,
            root=Path(sb.VIRTUAL_DEVICE_ID),
            target=Path(sb.VIRTUAL_DEVICE_ID),
            connected=True,
            display="agent",
            brightness=255,
        )
        snapshot = SimpleNamespace(statuses=[], collected_at=datetime.now(timezone.utc))
        for settings in (purple_settings(), purple_settings().with_rainbow_theme(True)):
            target = SimpleNamespace(
                settings=settings,
                closed_lid_awake=SimpleNamespace(last_error=None),
                status_bar_devices=lambda: [notch],
            )
            menu = sb.build_menu(snapshot, sb.STATE_IDLE, target)
            item = menu.itemWithTitle_("Rainbow Theme")
            self.assertIsNotNone(item)
            self.assertEqual(item.action(), "toggleRainbowTheme:")
            self.assertEqual(item.state(), 1 if settings.rainbow_theme_enabled else 0)
            notch_submenu = menu.itemWithTitle_(sb.VIRTUAL_DEVICE_NAME).submenu()
            self.assertIsNone(notch_submenu.itemWithTitle_("Rainbow Theme"))


if __name__ == "__main__":
    unittest.main()
