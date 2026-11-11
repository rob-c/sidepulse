"""Music spectrum: analysis, LED frames, audio decoding, and the menu-bar app."""

from __future__ import annotations

import cmath
import colorsys
import json
import math
import random
import struct
import tempfile
import threading
import time
import unittest
from array import array
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sidepulse import music_spectrum as ms
from sidepulse.device_writer import validate_led_text
from sidepulse.led_status import apply_brightness
from sidepulse.models import AgentMode
from sidepulse.settings import AgentMonitorSettings, load_settings, save_settings
from sidepulse.system_audio import (
    ERROR_USER_DECLINED,
    PERMISSION_HINT,
    _error_text,
    mono_from_pcm,
)


def tone(freq: float, amplitude: float = 0.5, count: int = 2048, offset: int = 0) -> list[float]:
    return [
        amplitude * math.sin(2.0 * math.pi * freq * (index + offset) / ms.SAMPLE_RATE)
        for index in range(count)
    ]


def settle(analyzer: ms.SpectrumAnalyzer, samples_for_step, steps: int = 20, start: float = 0.0):
    now = start
    frame = ms.SILENT_FRAME
    for step in range(steps):
        now += 0.033
        analyzer.push(samples_for_step(step), now)
        frame = analyzer.analyze(now)
    return frame, now


def hue(color) -> float:
    return colorsys.rgb_to_hsv(*color)[0]


class AnalyzerTests(unittest.TestCase):
    def test_bands_are_octave_spaced_and_cover_the_spectrum(self):
        edges = ms.band_edges()
        self.assertEqual(len(edges), ms.BAND_COUNT + 1)
        self.assertAlmostEqual(edges[0], ms.LOWEST_HZ)
        self.assertAlmostEqual(edges[-1], ms.HIGHEST_HZ)
        bins = ms.band_bins(ms.SAMPLE_RATE, ms.WINDOW_SIZE)
        self.assertEqual(len(bins), ms.BAND_COUNT)
        for previous, current in zip(bins, bins[1:]):
            self.assertGreater(len(current), 0)
            self.assertLessEqual(previous.stop, current.start)

    def test_fft_matches_a_direct_transform(self):
        values = [random.uniform(-1, 1) for _ in range(16)]
        expected = [
            sum(values[n] * cmath.exp(-2j * math.pi * k * n / 16) for n in range(16))
            for k in range(16)
        ]
        for got, want in zip(ms.make_fft(16)(values), expected):
            self.assertAlmostEqual(got, want, places=9)
        with self.assertRaises(ValueError):
            ms.make_fft(12)

    def test_tones_light_their_own_band(self):
        for freq, band in ((70, 0), (300, 2), (1500, 4), (6000, 6), (13000, 7)):
            with self.subTest(freq=freq):
                frame, _ = settle(ms.SpectrumAnalyzer(), lambda step: tone(freq, offset=step * 1600))
                self.assertTrue(frame.active)
                self.assertGreater(frame.levels[band], 0.9)
                others = [level for index, level in enumerate(frame.levels) if index != band]
                self.assertLess(max(others), 0.5)

    def test_quiet_music_still_fills_the_display(self):
        frame, _ = settle(
            ms.SpectrumAnalyzer(), lambda step: tone(1500, amplitude=0.01, offset=step * 1600)
        )
        self.assertGreater(frame.levels[4], 0.9)

    def test_volume_follows_sudden_drops(self):
        analyzer = ms.SpectrumAnalyzer()
        loud, now = settle(analyzer, lambda step: tone(440, 0.5, offset=step * 1600))
        quiet, _ = settle(analyzer, lambda step: tone(440, 0.02, offset=step * 1600), steps=6, start=now)
        self.assertGreater(loud.volume, 0.9)
        self.assertLess(quiet.volume, loud.volume - 0.4)

    def test_silence_is_not_music(self):
        analyzer = ms.SpectrumAnalyzer()
        analyzer.push([0.0] * 2048, 1.0)
        self.assertFalse(analyzer.analyze(1.0).active)

        frame, now = settle(analyzer, lambda step: tone(440, offset=step * 1600), start=1.0)
        self.assertTrue(frame.active)
        held = analyzer.analyze(now + ms.SILENCE_HOLD_SECONDS - 0.1)
        self.assertTrue(held.active)
        released = analyzer.analyze(now + ms.SILENCE_HOLD_SECONDS + 0.5)
        self.assertFalse(released.active)
        self.assertLess(max(released.levels), max(frame.levels))

    def test_reset_forgets_the_music(self):
        analyzer = ms.SpectrumAnalyzer()
        _, now = settle(analyzer, lambda step: tone(440, offset=step * 1600))
        analyzer.reset()
        self.assertEqual(analyzer.analyze(now), ms.SILENT_FRAME)


class FrameTests(unittest.TestCase):
    def frame(self, levels, volume=0.5, active=True):
        return ms.SpectrumFrame(tuple(levels), volume, active)

    def test_spectrum_runs_red_to_violet_at_band_brightness(self):
        colors = ms.spectrum_colors(self.frame([1.0] * 8))
        hues = [hue(color) for color in colors]
        self.assertAlmostEqual(hues[0], 0.0)
        self.assertAlmostEqual(hues[-1], ms.SPECTRUM_HUE_SPAN)
        self.assertEqual(hues, sorted(hues))
        self.assertAlmostEqual(max(colors[0]), 1.0)

        dim = ms.spectrum_colors(self.frame([0.0] * 8))
        self.assertAlmostEqual(max(dim[0]), ms.LEVEL_FLOOR)
        self.assertEqual(max(ms.spectrum_colors(self.frame([0.0] * 8, active=False))[0]), 0.0)

    def test_dot_pulses_with_volume_and_cycles_hue(self):
        loud = ms.pulse_color(self.frame([0] * 8, volume=1.0), 0.0)
        quiet = ms.pulse_color(self.frame([0] * 8, volume=0.1), 0.0)
        self.assertGreater(max(loud), max(quiet))
        later = ms.pulse_color(self.frame([0] * 8, volume=1.0), ms.DOT_HUE_CYCLE_SECONDS / 3)
        self.assertAlmostEqual(hue(later), 1 / 3, places=2)

        colors = ms.frame_colors(self.frame([1.0] * 8), 2, 0.0)
        self.assertEqual(len(colors), 2)
        self.assertEqual(colors[0], colors[1])
        self.assertEqual(len(ms.frame_colors(self.frame([1.0] * 8), 8, 0.0)), 8)

    def test_device_programs_are_valid_and_fit(self):
        spectrum = ms.device_program(ms.spectrum_colors(self.frame([1.0, 0.5, 0, 0, 0, 0, 0.2, 1.0])))
        self.assertTrue(spectrum.startswith("0:#FF0000 70ms linear; 1:"))
        self.assertEqual(spectrum.count(";"), 7)
        dot = ms.device_program([(1.0, 0.0, 0.0)] * 2)
        self.assertEqual(dot, "#FF0000 70ms linear")
        for program in (spectrum, dot):
            dimmed = apply_brightness(program, 100)
            validate_led_text(dimmed)
            self.assertLess(len(dimmed.encode()), 512)


class WriterTests(unittest.TestCase):
    def test_writes_only_the_newest_frame_per_device(self):
        written = []
        gate = threading.Event()

        def write(program, target):
            gate.wait(2)
            written.append((target, program))

        writer = ms.LatestProgramWriter(write)
        self.addCleanup(writer.stop)
        pro, dot = Path("/pro"), Path("/dot")
        writer.submit(pro, "frame 1")
        time.sleep(0.05)  # frame 1 is now being written
        for index in range(2, 6):
            writer.submit(pro, f"frame {index}")
        writer.submit(dot, "dot frame")
        gate.set()
        self.assertTrue(writer.wait_idle(2))
        self.assertEqual(
            [program for target, program in written if target == pro],
            ["frame 1", "frame 5"],
        )
        self.assertIn((dot, "dot frame"), written)

        writer.submit(pro, "frame 5")  # unchanged: not rewritten
        self.assertTrue(writer.wait_idle(1))
        self.assertEqual(len(written), 3)
        writer.forget()
        writer.submit(pro, "frame 5")
        self.assertTrue(writer.wait_idle(1))
        self.assertEqual(len(written), 4)

    def test_reports_write_errors_and_keeps_going(self):
        errors = []
        writer = ms.LatestProgramWriter(
            lambda program, target: (_ for _ in ()).throw(OSError("unplugged")),
            on_error=lambda target, error: errors.append((target, str(error))),
        )
        self.addCleanup(writer.stop)
        writer.submit(Path("/pro"), "frame")
        self.assertTrue(writer.wait_idle(1))
        self.assertEqual(errors, [(Path("/pro"), "unplugged")])


class SystemAudioDecodeTests(unittest.TestCase):
    def test_mixes_interleaved_and_planar_stereo_to_mono(self):
        interleaved = struct.pack("<6f", 0.2, 0.4, -0.2, -0.4, 1.0, 0.0)
        self.assertEqual(
            list(mono_from_pcm(interleaved, frames=3, channels=2, interleaved=True)),
            [array("f", [0.3])[0], array("f", [-0.3])[0], 0.5],
        )
        planar = struct.pack("<6f", 0.2, -0.2, 1.0, 0.4, -0.4, 0.0)
        self.assertEqual(
            list(mono_from_pcm(planar, frames=3, channels=2, interleaved=False)),
            list(mono_from_pcm(interleaved, frames=3, channels=2, interleaved=True)),
        )
        mono = struct.pack("<2f", 0.25, -0.5)
        self.assertEqual(list(mono_from_pcm(mono, frames=2, channels=1, interleaved=True)), [0.25, -0.5])
        sixteen = struct.pack("<2h", 16384, -32768)
        self.assertEqual(
            list(mono_from_pcm(sixteen, frames=2, channels=1, interleaved=True, is_float=False, bits=16)),
            [0.5, -1.0],
        )

    def test_reads_a_core_media_sample_buffer(self):
        try:
            import CoreAudio
            import CoreMedia as CM
        except ImportError as exc:
            self.skipTest(str(exc))
        from sidepulse.system_audio import mono_from_sample_buffer

        flags = (
            CoreAudio.kAudioFormatFlagIsFloat
            | CoreAudio.kAudioFormatFlagIsPacked
            | CoreAudio.kAudioFormatFlagIsNonInterleaved
        )
        description = CoreAudio.AudioStreamBasicDescription(
            mSampleRate=48000.0, mFormatID=CoreAudio.kAudioFormatLinearPCM, mFormatFlags=flags,
            mBytesPerPacket=4, mFramesPerPacket=1, mBytesPerFrame=4, mChannelsPerFrame=2,
            mBitsPerChannel=32,
        )
        status, format_description = CM.CMAudioFormatDescriptionCreate(
            None, description, 0, None, 0, None, None, None
        )
        self.assertEqual(status, 0)
        payload = struct.pack("<4f", 0.5, 1.0, -0.5, 0.0)  # left plane, then right
        status, block = CM.CMBlockBufferCreateWithMemoryBlock(
            None, None, len(payload), None, None, 0, len(payload),
            CM.kCMBlockBufferAssureMemoryNowFlag, None,
        )
        CM.CMBlockBufferReplaceDataBytes(payload, block, 0, len(payload))
        status, sample_buffer = CM.CMAudioSampleBufferCreateReadyWithPacketDescriptions(
            None, block, format_description, 2, CM.CMTimeMake(0, 48000), None, None
        )
        self.assertEqual(status, 0)
        self.assertEqual(list(mono_from_sample_buffer(sample_buffer)), [0.0, 0.5])

    def test_permission_errors_explain_what_to_allow(self):
        declined = SimpleNamespace(code=lambda: ERROR_USER_DECLINED)
        self.assertEqual(_error_text(declined), PERMISSION_HINT)
        self.assertIn("Screen & System Audio Recording", PERMISSION_HINT)
        other = SimpleNamespace(code=lambda: 1, localizedDescription=lambda: "Display went away")
        self.assertEqual(_error_text(other), "Display went away")


class MusicSpectrumSettingTests(unittest.TestCase):
    def test_off_by_default_and_saved(self):
        self.assertFalse(AgentMonitorSettings().music_spectrum_enabled)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            save_settings(AgentMonitorSettings().with_music_spectrum(True), path)
            self.assertIs(json.loads(path.read_text())["music_spectrum"], True)
            self.assertTrue(load_settings(path).music_spectrum_enabled)


try:
    from sidepulse import status_bar as sb
except (ImportError, SystemExit) as exc:  # AppKit and PyObjC are macOS-only.
    sb = None
    STATUS_BAR_SKIP = str(exc)

PLAYING = ms.SpectrumFrame((1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0, 0.5), 0.7, True)


class MusicSpectrumStatusBarTests(unittest.TestCase):
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

    def device(self, folder: str, *, display="agent", device_id=None, remote_link=None, brightness=255):
        root = self.root / folder
        return sb.StatusBarDevice(
            device_id=device_id or str(root),
            name=folder,
            root=root,
            target=root / "LEDS.LED",
            connected=True,
            display=display,
            brightness=brightness,
            remote_link=remote_link,
        )

    def controller(self, devices):
        controller = sb.StatusBarController.alloc().init()
        controller.settings = AgentMonitorSettings(
            virtual_status_device_enabled=True, music_spectrum_enabled=True
        )
        controller.leds_enabled = True
        controller.virtual_status_device = Mock()
        controller.spectrum_writer = Mock()
        controller.spectrum_analyzer = Mock()
        controller.spectrum_analyzer.analyze.return_value = PLAYING
        patcher = patch.object(sb.StatusBarController, "status_bar_devices", return_value=devices)
        patcher.start()
        self.addCleanup(patcher.stop)
        return controller

    def test_tick_paints_spectrum_on_pro_and_notch_and_volume_on_dot(self):
        pro = self.device("SidePulsePro", brightness=128)
        dot = self.device("PulseDot")
        custom = self.device("OtherPro", display=sb.LED_DISPLAY_CUSTOM)
        phone = self.device("phone", remote_link=SimpleNamespace(link_id="p"))
        notch = self.device("notch", device_id=sb.VIRTUAL_DEVICE_ID)
        controller = self.controller([pro, dot, custom, phone, notch])

        controller.musicSpectrumTick_(None)

        self.assertTrue(controller.spectrum_driving)
        controller.virtual_status_device.set_colors.assert_called_once_with(
            ms.spectrum_colors(PLAYING), notch.brightness
        )
        submitted = {call.args[0]: call.args[1] for call in controller.spectrum_writer.submit.call_args_list}
        self.assertEqual(set(submitted), {pro.target, dot.target})
        self.assertTrue(submitted[pro.target].startswith("brightness 128\n0:#"))
        self.assertEqual(submitted[pro.target].count(";"), 7)
        self.assertRegex(submitted[dot.target], r"^#[0-9A-F]{6} 70ms linear$")

        # The notch repaints every tick; hardware is paced to 20 fps.
        controller.musicSpectrumTick_(None)
        self.assertEqual(controller.virtual_status_device.set_colors.call_count, 2)
        self.assertEqual(controller.spectrum_writer.submit.call_count, 2)

    def test_alerts_lid_hold_and_animations_take_over(self):
        for blocker in ("alert", "lid", "animation"):
            with self.subTest(blocker=blocker):
                controller = self.controller([self.device("SidePulsePro")])
                controller.last_snapshot = SimpleNamespace(aggregate=SimpleNamespace(mode=AgentMode.WORKING))
                controller.musicSpectrumTick_(None)
                self.assertTrue(controller.spectrum_driving)
                if blocker == "alert":
                    controller.active_alerts = [object()]
                elif blocker == "lid":
                    controller.lid_closed_led_hold_active = True
                else:
                    controller.led_animation_until_monotonic = time.monotonic() + 30
                with patch.object(sb.StatusBarController, "sync_leds") as sync:
                    controller.musicSpectrumTick_(None)
                self.assertFalse(controller.spectrum_driving)
                controller.spectrum_writer.forget.assert_called_once_with()
                sync.assert_called_once()

    def test_silence_hands_the_leds_back(self):
        controller = self.controller([self.device("SidePulsePro")])
        controller.musicSpectrumTick_(None)
        controller.spectrum_analyzer.analyze.return_value = ms.SILENT_FRAME
        controller.last_snapshot = SimpleNamespace(aggregate=SimpleNamespace(mode=AgentMode.IDLE_READY))
        with patch.object(sb.StatusBarController, "sync_leds") as sync:
            controller.musicSpectrumTick_(None)
        self.assertFalse(controller.spectrum_driving)
        self.assertEqual(sync.call_args.args[0], AgentMode.IDLE_READY)

    def test_normal_sync_leaves_hardware_and_notch_alone_while_playing(self):
        pro = self.device("SidePulsePro")
        pro.root.mkdir()
        phone = self.device("phone", remote_link=SimpleNamespace(link_id="p"))
        notch = self.device("notch", device_id=sb.VIRTUAL_DEVICE_ID)
        controller = self.controller([pro, phone, notch])
        controller.spectrum_driving = True
        with patch.object(sb, "send_ios_program") as send:
            controller.sync_leds_now(AgentMode.WORKING, None, sb.LED_DISPLAY_AGENT)
            controller.sync_virtual_status_device(AgentMode.WORKING, None)
        self.assertFalse(pro.target.exists())
        send.assert_called_once()  # linked phones keep agent status
        controller.virtual_status_device.set_program.assert_not_called()

    def test_checkbox_and_menu_toggle_start_and_stop_it(self):
        controller = sb.StatusBarController.alloc().init()
        controller.settings = AgentMonitorSettings()
        controller.settings_window = sb.build_settings_window(controller)
        self.addCleanup(controller.settings_window.close)
        checkbox = controller.settings_buttons["music_spectrum"]
        with (
            patch.object(sb, "save_settings") as save,
            patch.object(sb.StatusBarController, "refresh_"),
            patch.object(sb.StatusBarController, "start_music_spectrum") as start,
            patch.object(sb.StatusBarController, "stop_music_spectrum") as stop,
        ):
            checkbox.setState_(sb.NSOnState)
            controller.setMusicSpectrumFromCheckbox_(checkbox)
            start.assert_called_once_with()
            self.assertTrue(save.call_args.args[0].music_spectrum_enabled)
            controller.toggleMusicSpectrum_(None)
            stop.assert_called_once_with()
            self.assertFalse(controller.settings.music_spectrum_enabled)
            self.assertEqual(checkbox.state(), sb.NSOffState)

    def test_start_reports_missing_capture_support(self):
        controller = sb.StatusBarController.alloc().init()
        controller.settings = AgentMonitorSettings(music_spectrum_enabled=True)
        capture = Mock()
        capture.start.side_effect = sb.SystemAudioUnavailable("needs macOS 13")
        controller.spectrum_capture = capture
        self.addCleanup(lambda: controller.stop_music_spectrum(restore=False))
        controller.start_music_spectrum()
        self.assertEqual(controller.spectrum_error, "needs macOS 13")
        self.assertIsNone(controller.spectrum_timer)

    def test_menu_toggle_shows_state_and_errors(self):
        snapshot = SimpleNamespace(statuses=[], collected_at=datetime.now(timezone.utc))
        target = SimpleNamespace(
            settings=AgentMonitorSettings(music_spectrum_enabled=True),
            closed_lid_awake=SimpleNamespace(last_error=None),
            status_bar_devices=lambda: [],
            spectrum_error=PERMISSION_HINT,
        )
        menu = sb.build_menu(snapshot, sb.STATE_IDLE, target)
        item = menu.itemWithTitle_("Music Spectrum")
        self.assertIsNotNone(item)
        self.assertEqual(item.action(), "toggleMusicSpectrum:")
        self.assertEqual(item.state(), 1)
        titles = [str(entry.title()) for entry in menu.itemArray()]
        self.assertTrue(any(title.startswith("Music spectrum: Allow SidePulse") for title in titles))

        target.settings = AgentMonitorSettings()
        menu = sb.build_menu(snapshot, sb.STATE_IDLE, target)
        self.assertEqual(menu.itemWithTitle_("Music Spectrum").state(), 0)
        self.assertFalse(any(str(e.title()).startswith("Music spectrum:") for e in menu.itemArray()))


if __name__ == "__main__":
    unittest.main()
