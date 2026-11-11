"""Rainbow theme for the SidePulse Notch and the animation previews."""

from __future__ import annotations

import colorsys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sidepulse.models import AgentMode
from sidepulse.settings import AgentMonitorSettings, load_settings, save_settings

try:
    from sidepulse import virtual_device as vd
except (ImportError, SystemExit) as exc:  # AppKit and PyObjC are macOS-only.
    vd = None
    VIRTUAL_DEVICE_SKIP = str(exc)


def hue_of(color) -> float:
    return colorsys.rgb_to_hsv(*color[:3])[0]


def hue_distance(first: float, second: float) -> float:
    gap = abs(first - second) % 1.0
    return min(gap, 1.0 - gap)


WHITE = (1.0, 1.0, 1.0, 1.0)


class RainbowColorTests(unittest.TestCase):
    def setUp(self):
        if vd is None:
            self.skipTest(VIRTUAL_DEVICE_SKIP)

    def test_keeps_each_led_brightness_and_alpha(self):
        # One frame of a sweep: dark LEDs, a rising edge, the peak, and a dim glow.
        colors = [
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.2, 0.25, 0.25),
            (0.0, 0.898, 1.0, 1.0),
            (0.6, 0.0, 0.6, 0.6),
            (2 / 255, 2 / 255, 4 / 255, 4 / 255),
            (0.95, 0.22, 0.1, 0.95),
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.227, 0.0, 1.0),
        ]
        rainbow = vd.rainbow_led_colors(colors, 12.3)

        self.assertEqual(len(rainbow), len(colors))
        for before, after in zip(colors, rainbow):
            self.assertAlmostEqual(max(after[:3]), max(before[:3]))
            self.assertEqual(after[3], before[3])
        self.assertEqual(rainbow[0], (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(rainbow[6], (0.0, 0.0, 0.0, 0.0))

    def test_spreads_one_full_rainbow_across_the_leds(self):
        hues = [hue_of(color) for color in vd.rainbow_led_colors([WHITE] * 8, 0.0)]

        for index in range(7):
            self.assertAlmostEqual(hue_distance(hues[index], hues[index + 1]), 1 / 8)
        self.assertAlmostEqual(hue_distance(hues[0], hues[4]), 0.5)

    def test_colors_flow_rightwards_over_time(self):
        step = vd.RAINBOW_CYCLE_SECONDS / 8
        before = vd.rainbow_led_colors([WHITE] * 8, 5.0)
        after = vd.rainbow_led_colors([WHITE] * 8, 5.0 + step)

        for index in range(1, 8):
            self.assertAlmostEqual(
                hue_distance(hue_of(after[index]), hue_of(before[index - 1])), 0.0
            )
        cycled = vd.rainbow_led_colors([WHITE] * 8, 5.0 + vd.RAINBOW_CYCLE_SECONDS)
        for first, second in zip(before, cycled):
            self.assertAlmostEqual(hue_distance(hue_of(first), hue_of(second)), 0.0)


class RainbowViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if vd is None:
            raise unittest.SkipTest(VIRTUAL_DEVICE_SKIP)
        from AppKit import NSApplication

        NSApplication.sharedApplication()

    def make_view(self):
        return vd.VirtualLedView.alloc().initWithFrame_(((0, 0), (220, 37)))

    def test_view_recolors_animations_only_when_enabled(self):
        view = self.make_view()
        view.setProgram_("#00E5FF")
        if view.wasm_controller is None:
            self.skipTest(f"LED engine unavailable: {view.wasm_error}")

        self.assertFalse(view.rainbow)
        plain = view._display_colors()
        self.assertEqual({color[:3] for color in plain}, {(0.0, 229 / 255, 1.0)})

        view.setRainbow_(True)
        rainbow = view._display_colors()
        self.assertEqual(len({round(hue_of(color), 3) for color in rainbow}), 8)
        for color in rainbow:
            self.assertAlmostEqual(max(color[:3]), 1.0)

    def test_battery_levels_keep_their_colors(self):
        view = self.make_view()
        view.setRainbow_(True)
        view.setBatteryPercent_brightness_(15, 255)

        self.assertEqual(view._display_colors(), view.fixed_colors)

    def test_notch_applies_the_theme_to_a_window_built_later(self):
        device = vd.VirtualStatusDevice.alloc().init()
        device.set_rainbow(True)
        device._build_window()
        self.addCleanup(device.window.close)

        self.assertTrue(device.view.rainbow)
        device.set_rainbow(False)
        self.assertFalse(device.view.rainbow)


class RainbowSettingTests(unittest.TestCase):
    def test_off_by_default_and_saved_with_settings(self):
        self.assertFalse(AgentMonitorSettings().rainbow_theme_enabled)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            save_settings(AgentMonitorSettings().with_rainbow_theme(True), path)
            self.assertIs(json.loads(path.read_text())["rainbow_theme"], True)
            self.assertTrue(load_settings(path).rainbow_theme_enabled)

            save_settings(load_settings(path).with_rainbow_theme(False), path)
            self.assertFalse(load_settings(path).rainbow_theme_enabled)


class RainbowStatusBarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from sidepulse import status_bar
        except (ImportError, SystemExit) as exc:
            raise unittest.SkipTest(str(exc))
        from AppKit import NSApplication

        NSApplication.sharedApplication()
        cls.sb = status_bar

    def device(self, device_id: str, name: str):
        return self.sb.StatusBarDevice(
            device_id=device_id,
            name=name,
            root=Path(device_id),
            target=Path(device_id),
            connected=True,
            display="agent",
            brightness=255,
        )

    def notch(self):
        return self.device(self.sb.VIRTUAL_DEVICE_ID, self.sb.VIRTUAL_DEVICE_NAME)

    def test_notch_menu_toggle_shows_the_current_setting(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                target = SimpleNamespace(
                    settings=AgentMonitorSettings(
                        virtual_status_device_enabled=True,
                        rainbow_theme_enabled=enabled,
                    )
                )
                submenu = self.sb.build_device_menu_item(self.notch(), target).submenu()
                item = submenu.itemWithTitle_("Rainbow Theme")

                self.assertIsNotNone(item)
                self.assertEqual(item.action(), "toggleRainbowTheme:")
                self.assertIs(item.target(), target)
                self.assertEqual(item.state(), 1 if enabled else 0)

        hardware = self.device("/Volumes/SidePulsePro", "SidePulse Pro")
        target = SimpleNamespace(settings=AgentMonitorSettings())
        submenu = self.sb.build_device_menu_item(hardware, target).submenu()
        self.assertIsNone(submenu.itemWithTitle_("Rainbow Theme"))

    def test_checkbox_and_menu_toggle_save_and_recolor_previews(self):
        sb = self.sb
        controller = sb.StatusBarController.alloc().init()
        controller.settings = AgentMonitorSettings()
        controller.settings_window = sb.build_settings_window(controller)
        self.addCleanup(controller.settings_window.close)
        checkbox = controller.settings_buttons["rainbow_theme"]
        editor = sb.build_agent_animation_editor_window(controller)
        self.addCleanup(editor.close)

        def previews():
            for mode in sb.ANIMATION_UI_STATES:
                yield controller.settings_fields[f"agent_animation_preview_{mode}"]
                popup = controller.settings_fields[f"agent_animation_{mode}"]
                for index in range(popup.numberOfItems()):
                    view = popup.itemAtIndex_(index).view()
                    if view is not None:
                        yield view.animation_preview

        with (
            patch.object(sb, "save_settings") as save,
            patch.object(sb.StatusBarController, "refresh_") as refresh,
        ):
            checkbox.setState_(sb.NSOnState)
            controller.setRainbowThemeFromCheckbox_(checkbox)
            self.assertTrue(save.call_args.args[0].rainbow_theme_enabled)
            self.assertTrue(controller.settings.rainbow_theme_enabled)
            refresh.assert_called()
            self.assertTrue(all(preview.rainbow for preview in previews()))
            # The custom-animation editor shows true colors.
            self.assertFalse(controller.agent_animation_editor_preview.rainbow)

            controller.toggleRainbowTheme_(None)
            self.assertFalse(save.call_args.args[0].rainbow_theme_enabled)
            self.assertEqual(checkbox.state(), sb.NSOffState)
            self.assertFalse(any(preview.rainbow for preview in previews()))

    def test_notch_is_rainbow_for_agent_animations_but_not_battery(self):
        sb = self.sb
        controller = sb.StatusBarController.alloc().init()
        controller.settings = AgentMonitorSettings(
            virtual_status_device_enabled=True,
            rainbow_theme_enabled=True,
        )
        controller.virtual_status_device = Mock()
        notch = controller.virtual_status_device

        with (
            patch.object(sb.StatusBarController, "status_bar_devices", return_value=[self.notch()]),
            patch.object(
                sb.StatusBarController,
                "active_led_display_kind_for_device",
                side_effect=[sb.LED_DISPLAY_AGENT, sb.LED_DISPLAY_BATTERY],
            ),
            patch.object(sb, "program_for_battery", return_value="off"),
        ):
            controller.sync_virtual_status_device(AgentMode.WORKING, None)
            notch.set_rainbow.assert_called_with(True)
            notch.set_program.assert_called()

            controller.sync_virtual_status_device(AgentMode.WORKING, object())
            notch.set_rainbow.assert_called_with(False)
            notch.set_program.assert_called_with("off")


if __name__ == "__main__":
    unittest.main()
