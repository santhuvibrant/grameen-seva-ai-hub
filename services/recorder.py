"""Browser-side autonomous microphone component."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


_COMPONENT = components.declare_component(
    "autonomous_recorder",
    path=str(Path(__file__).resolve().parent.parent / "components" / "autonomous_recorder"),
)


def autonomous_recorder(
    *,
    active: bool,
    auto_start: bool,
    tts_audio: bytes | None,
    tts_token: int,
    resume_after_tts: bool,
    reset_token: int,
) -> dict[str, Any] | None:
    """Return a completed utterance or a one-time recorder event."""
    audio = base64.b64encode(tts_audio).decode("ascii") if tts_audio else ""
    return _COMPONENT(
        active=active,
        auto_start=auto_start,
        tts_audio=audio,
        tts_token=tts_token,
        resume_after_tts=resume_after_tts,
        reset_token=reset_token,
        key="autonomous-recorder",
        default=None,
    )
