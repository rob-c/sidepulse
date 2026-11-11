"""Capture the Mac's sound output with ScreenCaptureKit (macOS 13 or later).

The first start makes macOS ask for Screen & System Audio Recording permission. Only
the audio is used: the stream's video is a 2x2-pixel frame once a second, which is
ignored, and SidePulse's own process is excluded from the capture.

The macOS frameworks are imported when capture starts, so this module also imports
on other platforms.
"""

from __future__ import annotations

import struct
import threading
import time
from array import array
from typing import Any, Callable

SAMPLE_RATE = 48_000
CHANNEL_COUNT = 2
# AudioStreamBasicDescription flags (CoreAudioBaseTypes.h).
FORMAT_FLAG_IS_FLOAT = 1 << 0
FORMAT_FLAG_IS_NON_INTERLEAVED = 1 << 5
# SCStreamErrorUserDeclined: the user has not allowed screen and audio recording.
ERROR_USER_DECLINED = -3801
PERMISSION_HINT = (
    "Allow SidePulse (it may be listed as Python) in System Settings > Privacy & "
    "Security > Screen & System Audio Recording, then turn Music Spectrum off and on."
)


class SystemAudioUnavailable(RuntimeError):
    pass


def mono_from_pcm(
    data: bytes,
    *,
    frames: int,
    channels: int,
    interleaved: bool,
    is_float: bool = True,
    bits: int = 32,
) -> array:
    """Mix PCM bytes from a capture buffer down to mono float samples."""
    if frames <= 0 or channels <= 0:
        return array("f")
    if is_float and bits == 32:
        samples = array("f")
        samples.frombytes(data[: (len(data) // 4) * 4])
    elif not is_float and bits == 16:
        count = len(data) // 2
        samples = array("f", (value / 32768.0 for value in struct.unpack(f"<{count}h", data[: count * 2])))
    else:
        raise SystemAudioUnavailable(f"Unsupported audio format: {bits}-bit {'float' if is_float else 'integer'}")
    frames = min(frames, len(samples) // channels)
    if channels == 1:
        return samples[:frames]
    if interleaved:
        planes = [samples[channel : frames * channels : channels] for channel in range(channels)]
    else:
        planes = [samples[channel * frames : (channel + 1) * frames] for channel in range(channels)]
    scale = 1.0 / channels
    return array("f", (sum(values) * scale for values in zip(*planes)))


def mono_from_sample_buffer(sample_buffer: Any) -> array:
    """Mono samples from a ScreenCaptureKit audio CMSampleBuffer."""
    import CoreMedia

    description = CoreMedia.CMSampleBufferGetFormatDescription(sample_buffer)
    if description is None:
        return array("f")
    stream = CoreMedia.CMAudioFormatDescriptionGetStreamBasicDescription(description)
    flags = int(_field(stream, "mFormatFlags", 2))
    channels = int(_field(stream, "mChannelsPerFrame", 6))
    bits = int(_field(stream, "mBitsPerChannel", 7))
    block = CoreMedia.CMSampleBufferGetDataBuffer(sample_buffer)
    if block is None:
        return array("f")
    length = CoreMedia.CMBlockBufferGetDataLength(block)
    status, data = CoreMedia.CMBlockBufferCopyDataBytes(block, 0, length, None)
    if status != 0 or data is None:
        return array("f")
    return mono_from_pcm(
        bytes(data),
        frames=int(CoreMedia.CMSampleBufferGetNumSamples(sample_buffer)),
        channels=channels,
        interleaved=not flags & FORMAT_FLAG_IS_NON_INTERLEAVED,
        is_float=bool(flags & FORMAT_FLAG_IS_FLOAT),
        bits=bits,
    )


def _field(struct_value: Any, name: str, index: int) -> Any:
    value = getattr(struct_value, name, None)
    return struct_value[index] if value is None else value


_OUTPUT_CLASS = None


def _output_class():
    """The ObjC stream output class, defined once per process."""
    global _OUTPUT_CLASS
    if _OUTPUT_CLASS is not None:
        return _OUTPUT_CLASS
    import objc
    import ScreenCaptureKit
    from Foundation import NSObject

    class SidePulseSystemAudioOutput(
        NSObject,
        protocols=[objc.protocolNamed("SCStreamOutput"), objc.protocolNamed("SCStreamDelegate")],
    ):
        def initWithCapture_(self, capture):
            self = objc.super(SidePulseSystemAudioOutput, self).init()
            if self is not None:
                self.capture = capture
            return self

        def stream_didOutputSampleBuffer_ofType_(self, _stream, sample_buffer, output_type):
            if output_type != ScreenCaptureKit.SCStreamOutputTypeAudio:
                return
            try:
                samples = mono_from_sample_buffer(sample_buffer)
            except Exception as exc:
                self.capture._fail(f"Could not read system audio: {exc}")
                return
            self.capture._deliver(samples)

        def stream_didStopWithError_(self, _stream, error):
            self.capture._fail(_error_text(error))

    _OUTPUT_CLASS = SidePulseSystemAudioOutput
    return _OUTPUT_CLASS


def _error_text(error: Any) -> str:
    if error is None:
        return "System audio capture stopped."
    try:
        if int(error.code()) == ERROR_USER_DECLINED:
            return PERMISSION_HINT
        return str(error.localizedDescription())
    except Exception:
        return str(error)


class SystemAudioCapture:
    """Streams the Mac's sound output, mixed to mono, to ``on_samples``.

    ``on_samples(samples, monotonic_time)`` and ``on_error(message)`` are called on a
    background queue. ``start`` returns at once; failures arrive through ``on_error``.
    """

    def __init__(
        self,
        on_samples: Callable[[array, float], None],
        *,
        on_error: Callable[[str], None] | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.on_samples = on_samples
        self.on_error = on_error
        self.sample_rate = sample_rate
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._stream = None
        self._output = None
        self._queue = None
        self._generation = 0

    @property
    def running(self) -> bool:
        with self._lock:
            return self._stream is not None

    def start(self) -> None:
        try:
            import dispatch
            import ScreenCaptureKit
        except ImportError as exc:
            raise SystemAudioUnavailable(
                f"System audio capture needs ScreenCaptureKit (macOS 13 or later): {exc}"
            ) from exc
        with self._lock:
            if self._stream is not None:
                return
            self._generation += 1
            generation = self._generation
            if self._queue is None:
                self._queue = dispatch.dispatch_queue_create(b"io.sidepulse.system-audio", None)
        self.last_error = None

        def received_content(content, error):
            if error is not None or content is None:
                self._fail(_error_text(error))
                return
            try:
                self._start_stream(content, generation)
            except Exception as exc:
                self._fail(f"Could not start system audio capture: {exc}")

        ScreenCaptureKit.SCShareableContent.getShareableContentWithCompletionHandler_(
            received_content
        )

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            stream = self._stream
            self._stream = None
            self._output = None
        if stream is not None:
            try:
                stream.stopCaptureWithCompletionHandler_(lambda _error: None)
            except Exception:
                pass

    def _start_stream(self, content, generation: int) -> None:
        import CoreMedia
        import ScreenCaptureKit as SCK

        displays = content.displays()
        if not displays:
            raise SystemAudioUnavailable("No display is available to capture audio from.")
        content_filter = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
            displays[0], []
        )
        configuration = SCK.SCStreamConfiguration.alloc().init()
        configuration.setCapturesAudio_(True)
        configuration.setExcludesCurrentProcessAudio_(True)
        configuration.setSampleRate_(self.sample_rate)
        configuration.setChannelCount_(CHANNEL_COUNT)
        configuration.setWidth_(2)
        configuration.setHeight_(2)
        configuration.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, 1))

        output = _output_class().alloc().initWithCapture_(self)
        stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, configuration, output
        )
        # Screen frames are ignored, but without an output the system logs a warning
        # for each one it drops.
        for output_type in (SCK.SCStreamOutputTypeAudio, SCK.SCStreamOutputTypeScreen):
            added, error = stream.addStreamOutput_type_sampleHandlerQueue_error_(
                output, output_type, self._queue, None
            )
            if not added:
                raise SystemAudioUnavailable(_error_text(error))

        with self._lock:
            if generation != self._generation:
                return  # Stopped while permission or content was pending.
            self._stream = stream
            self._output = output

        def started(error):
            if error is not None:
                self._fail(_error_text(error))

        stream.startCaptureWithCompletionHandler_(started)

    def _deliver(self, samples: array) -> None:
        if samples:
            self.on_samples(samples, time.monotonic())

    def _fail(self, message: str) -> None:
        self.last_error = message
        with self._lock:
            self._stream = None
            self._output = None
        if self.on_error is not None:
            self.on_error(message)
