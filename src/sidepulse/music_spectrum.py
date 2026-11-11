"""Music spectrum: turn the Mac's sound output into LED colors.

SidePulse Pro and the SidePulse Notch show an eight-band rainbow spectrum, bass in red
on the left through treble in violet on the right, each LED as bright as its band.
SidePulse Dot pulses with the overall volume while its color cycles through the
rainbow. Everything here is plain Python; ``system_audio`` supplies the samples.
"""

from __future__ import annotations

import cmath
import colorsys
import math
import threading
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

SAMPLE_RATE = 48_000
WINDOW_SIZE = 2048  # 43 ms at 48 kHz; 23 Hz bins resolve the bass bands.
BAND_COUNT = 8
LOWEST_HZ = 50.0
HIGHEST_HZ = 16_000.0
# Bands light up across this many dB below the loudest recent band, so quiet and
# loud music both fill the display.
DYNAMIC_RANGE_DB = 42.0
VOLUME_RANGE_DB = 30.0
PEAK_DECAY_DB_PER_SECOND = 6.0
REFERENCE_FLOOR_DB = -60.0
SILENCE_DBFS = -70.0
# After this long without sound the LEDs go back to showing agent status.
SILENCE_HOLD_SECONDS = 2.5
ATTACK_SECONDS = 0.025
RELEASE_SECONDS = 0.22
# Red (bass) to violet (treble), stopping short of wrapping back to red.
SPECTRUM_HUE_SPAN = 0.78
DOT_HUE_CYCLE_SECONDS = 8.0
# Keep the rainbow faintly lit between beats while music plays.
LEVEL_FLOOR = 0.06
DEVICE_TRANSITION_MS = 70


@dataclass(frozen=True)
class SpectrumFrame:
    levels: tuple[float, ...]  # 0..1 per band, bass first
    volume: float  # 0..1
    active: bool  # sound has played within SILENCE_HOLD_SECONDS


SILENT_FRAME = SpectrumFrame((0.0,) * BAND_COUNT, 0.0, False)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def band_edges(
    band_count: int = BAND_COUNT,
    low_hz: float = LOWEST_HZ,
    high_hz: float = HIGHEST_HZ,
) -> list[float]:
    """Logarithmically spaced edges, so each band is roughly an octave wide."""
    ratio = high_hz / low_hz
    return [low_hz * ratio ** (index / band_count) for index in range(band_count + 1)]


def band_bins(sample_rate: int, window_size: int, band_count: int = BAND_COUNT) -> list[range]:
    """FFT bin ranges per band; every band gets at least one bin."""
    resolution = sample_rate / window_size
    edges = band_edges(band_count)
    ranges: list[range] = []
    previous_end = 1
    for low, high in zip(edges, edges[1:]):
        start = max(previous_end, int(math.floor(low / resolution)))
        end = max(start + 1, int(math.ceil(high / resolution)))
        end = min(end, window_size // 2)
        ranges.append(range(start, max(start + 1, end)))
        previous_end = end
    return ranges


def make_fft(size: int) -> Callable[[Sequence[float]], list[complex]]:
    """An iterative radix-2 FFT; about 1 ms for 2048 points in CPython."""
    if size < 2 or size & (size - 1):
        raise ValueError("FFT size must be a power of two")
    bits = size.bit_length() - 1
    reverse = [int(f"{index:0{bits}b}"[::-1], 2) for index in range(size)]
    twiddles = [cmath.exp(-2j * math.pi * k / size) for k in range(size // 2)]

    def fft(values: Sequence[float]) -> list[complex]:
        data = [complex(values[index]) for index in reverse]
        span = 2
        while span <= size:
            half = span // 2
            step = size // span
            for start in range(0, size, span):
                twiddle = 0
                for index in range(start, start + half):
                    product = twiddles[twiddle] * data[index + half]
                    data[index + half] = data[index] - product
                    data[index] += product
                    twiddle += step
            span *= 2
        return data

    return fft


def smooth(current: float, target: float, elapsed: float) -> float:
    seconds = ATTACK_SECONDS if target > current else RELEASE_SECONDS
    if elapsed <= 0.0:
        return current
    return current + (target - current) * (1.0 - math.exp(-elapsed / seconds))


class SpectrumAnalyzer:
    """Keeps the latest audio window and turns it into smoothed band levels.

    ``push`` is called from the capture thread, ``analyze`` from the frame timer.
    """

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        window_size: int = WINDOW_SIZE,
        band_count: int = BAND_COUNT,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.band_count = band_count
        self._lock = threading.Lock()
        self._ring = array("f", bytes(4 * window_size))
        self._write_index = 0
        self._last_sound_at: float | None = None
        self._fft = make_fft(window_size)
        self._window = [
            0.5 - 0.5 * math.cos(2.0 * math.pi * index / (window_size - 1))
            for index in range(window_size)
        ]
        # Full-scale sine in one bin reads about 0 dB.
        self._power_scale = 4.0 / (sum(self._window) ** 2)
        self._bins = band_bins(sample_rate, window_size, band_count)
        self._levels = [0.0] * band_count
        self._volume = 0.0
        self._band_reference_db = REFERENCE_FLOOR_DB
        self._volume_reference_db = REFERENCE_FLOOR_DB
        self._last_analyzed_at: float | None = None

    def push(self, samples: Sequence[float], now: float) -> None:
        if not samples:
            return
        loudest = max(abs(min(samples)), abs(max(samples)))
        chunk = array("f", samples[-self.window_size :])
        with self._lock:
            index = self._write_index
            first = min(len(chunk), self.window_size - index)
            self._ring[index : index + first] = chunk[:first]
            rest = len(chunk) - first
            if rest:
                self._ring[:rest] = chunk[first:]
            self._write_index = (index + len(chunk)) % self.window_size
            if loudest > 10 ** (SILENCE_DBFS / 20.0):
                self._last_sound_at = now

    def reset(self) -> None:
        with self._lock:
            self._ring = array("f", bytes(4 * self.window_size))
            self._write_index = 0
            self._last_sound_at = None
        self._levels = [0.0] * self.band_count
        self._volume = 0.0
        self._band_reference_db = REFERENCE_FLOOR_DB
        self._volume_reference_db = REFERENCE_FLOOR_DB
        self._last_analyzed_at = None

    def analyze(self, now: float) -> SpectrumFrame:
        with self._lock:
            start = self._write_index
            ordered = self._ring[start:] + self._ring[:start]
            last_sound_at = self._last_sound_at
        elapsed = 0.0 if self._last_analyzed_at is None else now - self._last_analyzed_at
        self._last_analyzed_at = now
        active = last_sound_at is not None and now - last_sound_at <= SILENCE_HOLD_SECONDS
        if not active:
            self._levels = [smooth(level, 0.0, elapsed) for level in self._levels]
            self._volume = smooth(self._volume, 0.0, elapsed)
            return SpectrumFrame(tuple(self._levels), self._volume, False)

        spectrum = self._fft([value * weight for value, weight in zip(ordered, self._window)])
        band_db = []
        for bins in self._bins:
            power = sum(abs(spectrum[index]) ** 2 for index in bins) * self._power_scale
            band_db.append(10.0 * math.log10(power + 1e-12))
        mean_square = sum(value * value for value in ordered) / len(ordered)
        volume_db = 10.0 * math.log10(mean_square + 1e-12)

        decay = PEAK_DECAY_DB_PER_SECOND * elapsed
        self._band_reference_db = max(
            max(band_db), self._band_reference_db - decay, REFERENCE_FLOOR_DB
        )
        self._volume_reference_db = max(
            volume_db, self._volume_reference_db - decay, REFERENCE_FLOOR_DB
        )
        band_floor = self._band_reference_db - DYNAMIC_RANGE_DB
        volume_floor = self._volume_reference_db - VOLUME_RANGE_DB
        self._levels = [
            smooth(level, clamp((db - band_floor) / DYNAMIC_RANGE_DB), elapsed)
            for level, db in zip(self._levels, band_db)
        ]
        self._volume = smooth(
            self._volume, clamp((volume_db - volume_floor) / VOLUME_RANGE_DB), elapsed
        )
        return SpectrumFrame(tuple(self._levels), self._volume, True)


def spectrum_colors(frame: SpectrumFrame) -> list[tuple[float, float, float]]:
    """One rainbow color per band, as bright as the band (0..1 channels)."""
    count = len(frame.levels)
    colors = []
    for index, level in enumerate(frame.levels):
        hue = SPECTRUM_HUE_SPAN * index / max(1, count - 1)
        value = LEVEL_FLOOR + (1.0 - LEVEL_FLOOR) * level if frame.active else level
        colors.append(colorsys.hsv_to_rgb(hue, 1.0, clamp(value)))
    return colors


def pulse_color(frame: SpectrumFrame, now: float) -> tuple[float, float, float]:
    """The Dot's color: the volume sets brightness while the hue cycles."""
    hue = (now / DOT_HUE_CYCLE_SECONDS) % 1.0
    value = LEVEL_FLOOR + (1.0 - LEVEL_FLOOR) * frame.volume if frame.active else frame.volume
    return colorsys.hsv_to_rgb(hue, 1.0, clamp(value))


def frame_colors(
    frame: SpectrumFrame,
    led_count: int,
    now: float,
) -> list[tuple[float, float, float]]:
    if led_count <= 2:
        return [pulse_color(frame, now)] * led_count
    return spectrum_colors(frame)[:led_count]


def rgb_hex(color: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(clamp(channel) * 255):02X}" for channel in color)


def device_program(
    colors: Sequence[tuple[float, float, float]],
    *,
    transition_ms: int = DEVICE_TRANSITION_MS,
) -> str:
    """An LEDS.LED line that eases every LED to its new color."""
    if colors and all(rgb_hex(color) == rgb_hex(colors[0]) for color in colors):
        return f"{rgb_hex(colors[0])} {transition_ms}ms linear"
    return "; ".join(
        f"{index}:{rgb_hex(color)} {transition_ms}ms linear"
        for index, color in enumerate(colors)
    )


class LatestProgramWriter:
    """Writes the newest program for each device on one background thread.

    Device writes can take longer than a frame; older frames are dropped rather than
    queued, so the LEDs never lag behind the music.
    """

    def __init__(
        self,
        write: Callable[[str, Path], object],
        *,
        on_error: Callable[[Path, Exception], None] | None = None,
    ) -> None:
        self._write = write
        self._on_error = on_error
        self._pending: dict[Path, str] = {}
        self._written: dict[Path, str] = {}
        self._condition = threading.Condition()
        self._stopped = False
        self._busy = False
        self._generation = 0
        self._thread = threading.Thread(
            target=self._run, name="sidepulse-spectrum-writer", daemon=True
        )
        self._thread.start()

    def submit(self, target: Path, program: str) -> None:
        with self._condition:
            if self._written.get(target) == program:
                self._pending.pop(target, None)
                return
            self._pending[target] = program
            self._condition.notify()

    def forget(self) -> None:
        """Drop pending frames, and rewrite every device on its next frame."""
        with self._condition:
            self._pending.clear()
            self._written.clear()
            self._generation += 1

    def wait_idle(self, timeout: float = 1.0) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: not self._pending and not self._busy, timeout
            )

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._pending.clear()
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._stopped)
                if self._stopped:
                    return
                batch = self._pending
                self._pending = {}
                self._busy = True
                generation = self._generation
            for target, program in batch.items():
                try:
                    self._write(program, target)
                except Exception as exc:
                    if self._on_error is not None:
                        self._on_error(target, exc)
                    continue
                with self._condition:
                    if generation == self._generation:
                        self._written[target] = program
            with self._condition:
                self._busy = False
                self._condition.notify_all()
