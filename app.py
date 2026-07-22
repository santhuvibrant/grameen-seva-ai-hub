"""
Grameen Seva AI Hub — Voice-first agricultural subsidy finder for science-fair kiosk.

Pipeline: Mic → Sarvam STT → Gemini 2.5 (Tavily + Firecrawl tools) → Metric cards → Sarvam TTS
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

import qrcode
import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Page config — wide desktop / kiosk layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Grameen Seva AI Hub",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Session state defaults
for key, default in {
    "transcript": "",
    "voice_response": "",
    "scheme_name": "",
    "equipment": "",
    "subsidy_percent": 0,
    "max_claim_inr": 0,
    "missing_criteria": "",
    "card_status": "idle",
    "tts_audio_bytes": None,
    "last_audio_hash": None,
    "replay_counter": 0,
}.items():
    st.session_state.setdefault(key, default)

# ---------------------------------------------------------------------------
# Custom CSS — high-contrast kiosk styling (readable from ~5 feet)
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #E8F5E9 0%, #FAFAFA 50%, #FFFFFF 100%);
    }
    .main-header {
        text-align: center;
        color: #1B5E20;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: 0.5px;
        margin: 0.5rem 0 0.25rem 0;
        line-height: 1.15;
    }
    .main-subtitle {
        text-align: center;
        color: #2E7D32;
        font-size: 1.35rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .panel-title {
        color: #2E7D32;
        font-size: 1.8rem;
        font-weight: 800;
        border-bottom: 4px solid #2E7D32;
        padding-bottom: 0.5rem;
        margin-bottom: 1.25rem;
    }
    .control-box {
        background: #FFFFFF;
        border: 2px solid #A5D6A7;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 6px 18px rgba(46, 125, 50, 0.12);
    }
    .transcript-box {
        background: #E3F2FD;
        border-left: 8px solid #1565C0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        font-size: 1.45rem;
        font-weight: 600;
        color: #0D47A1;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.25rem;
        margin: 1.25rem 0;
    }
    .metric-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        border-top: 6px solid #2E7D32;
    }
    .metric-card.warning {
        border-top-color: #F9A825;
        background: #FFFDE7;
    }
    .metric-label {
        font-size: 1.1rem;
        font-weight: 700;
        color: #546E7A;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.6rem;
        font-weight: 900;
        color: #1B5E20;
        line-height: 1.15;
    }
    .metric-value.highlight {
        font-size: 3.2rem;
        color: #2E7D32;
    }
    .scheme-banner {
        background: linear-gradient(90deg, #2E7D32, #43A047);
        color: white;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 1.25rem;
        text-align: center;
    }
    .missing-banner {
        background: #FFF3E0;
        border-left: 8px solid #EF6C00;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        font-size: 1.25rem;
        font-weight: 700;
        color: #E65100;
        margin-top: 1rem;
    }
    div[data-testid="stAudioInput"] label {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #2E7D32 !important;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="column"] label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    #MainMenu, footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDIAN_STATES = [
    "Telangana",
    "Andhra Pradesh",
    "Karnataka",
    "Tamil Nadu",
    "Maharashtra",
    "Gujarat",
    "Rajasthan",
    "Punjab",
    "Uttar Pradesh",
    "Bihar",
    "West Bengal",
    "Kerala",
    "Madhya Pradesh",
    "Odisha",
]

FARMER_CATEGORIES = [
    "Small/Marginal Farmer",
    "General Farmer",
    "SC/ST Farmer",
]

LANGUAGE_OPTIONS = {
    "Hindi (hi-IN)": "hi-IN",
    "Telugu (te-IN)": "te-IN",
    "Tamil (ta-IN)": "ta-IN",
}

SYSTEM_PROMPT = """You are Grameen Seva AI Hub, an expert assistant helping Indian farmers
find government agricultural subsidies and schemes.

Use your tools to search myscheme.gov.in and gov.in, then read promising pages for details.

Extract:
- equipment_or_input: what the farmer needs (e.g., drip kit, tractor, seeds)
- scheme_name: official scheme name
- subsidy_percent: numeric percentage (0 if unknown)
- max_claim_inr: maximum claimable amount in INR as integer (0 if unknown)
- missing_criteria: ONE missing detail blocking full eligibility, or null if complete
- voice_response: 3-5 sentence spoken summary in the farmer's selected language script

Respond with ONLY valid JSON (no markdown fences):
{
  "equipment_or_input": "...",
  "scheme_name": "...",
  "subsidy_percent": 60,
  "max_claim_inr": 120000,
  "missing_criteria": null,
  "voice_response": "...",
  "source_url": "https://..."
}

Be accurate. Only cite schemes found via tools. Never invent amounts or scheme names.
"""


# ---------------------------------------------------------------------------
# Secrets & cached clients
# ---------------------------------------------------------------------------


def get_secret(name: str) -> str | None:
    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError, TypeError):
        return None


def missing_secrets() -> list[str]:
    return [k for k in ("SARVAM_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY", "FIRECRAWL_API_KEY") if not get_secret(k)]


@st.cache_resource
def sarvam_client(api_key: str):
    from sarvamai import SarvamAI

    return SarvamAI(api_subscription_key=api_key)


@st.cache_resource
def gemini_client(api_key: str):
    return genai.Client(api_key=api_key)


@st.cache_resource
def tavily_client(api_key: str):
    from tavily import TavilyClient

    return TavilyClient(api_key=api_key)


@st.cache_resource
def firecrawl_client(api_key: str):
    from firecrawl import Firecrawl

    return Firecrawl(api_key=api_key)


# ---------------------------------------------------------------------------
# Sarvam STT & TTS
# ---------------------------------------------------------------------------


def transcribe(audio_bytes: bytes, language_code: str) -> str:
    """Send WAV bytes from st.audio_input to Sarvam saaras:v3."""
    client = sarvam_client(get_secret("SARVAM_API_KEY"))
    buffer = io.BytesIO(audio_bytes)
    result = client.speech_to_text.transcribe(
        file=buffer,
        model="saaras:v3",
        mode="transcribe",
        language_code=language_code,
    )
    return (result.transcript or "").strip()


def text_to_speech(text: str, language_code: str) -> bytes:
    """Convert agent summary to WAV bytes via Sarvam bulbul:v3."""
    client = sarvam_client(get_secret("SARVAM_API_KEY"))
    spoken = text[:2500] if len(text) > 2500 else text
    result = client.text_to_speech.convert(
        text=spoken,
        target_language_code=language_code,
        model="bulbul:v3",
        speaker="shubh",
    )
    raw = result.audios[0]
    return base64.b64decode(raw) if isinstance(raw, str) else raw


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


def search_schemes(query: str, state: str) -> str:
    """Search government subsidy schemes via Tavily (myscheme.gov.in / gov.in)."""
    client = tavily_client(get_secret("TAVILY_API_KEY"))
    scoped = f"{query} {state} agricultural subsidy site:myscheme.gov.in OR site:gov.in"
    response = client.search(
        query=scoped,
        search_depth="advanced",
        max_results=5,
        include_domains=["myscheme.gov.in", "gov.in"],
    )
    hits = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", "")[:1200],
            "score": r.get("score", 0),
        }
        for r in response.get("results", [])
    ]
    return json.dumps({"query": scoped, "results": hits}, ensure_ascii=False)


def get_scheme_details(url: str) -> str:
    """Scrape full scheme page content via Firecrawl."""
    client = firecrawl_client(get_secret("FIRECRAWL_API_KEY"))
    doc = client.scrape(url, formats=["markdown"])
    md = doc.markdown if hasattr(doc, "markdown") else doc.get("markdown", "")
    return (md[:8000] + "\n[truncated]") if len(md) > 8000 else md


# ---------------------------------------------------------------------------
# Gemini agent with manual function calling
# ---------------------------------------------------------------------------


def _tools() -> list[types.Tool]:
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="search_schemes",
                    description="Search Indian govt subsidy schemes on myscheme.gov.in and gov.in.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "query": types.Schema(type="STRING", description="Farmer need or product"),
                            "state": types.Schema(type="STRING", description="Indian state"),
                        },
                        required=["query", "state"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="get_scheme_details",
                    description="Read full markdown content from a scheme webpage URL.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={"url": types.Schema(type="STRING", description="Scheme page URL")},
                        required=["url"],
                    ),
                ),
            ]
        )
    ]


def _run_tool(name: str, args: dict[str, Any], state: str) -> str:
    if name == "search_schemes":
        return search_schemes(args.get("query", ""), args.get("state", state))
    if name == "get_scheme_details":
        return get_scheme_details(args.get("url", ""))
    return json.dumps({"error": f"Unknown tool: {name}"})


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {
        "equipment_or_input": "",
        "scheme_name": "Information Pending",
        "subsidy_percent": 0,
        "max_claim_inr": 0,
        "missing_criteria": "Could not parse agent response",
        "voice_response": text[:400] if text else "क्षमा करें, जानकारी प्राप्त नहीं हो सकी।",
        "source_url": "",
    }


def run_agent(transcript: str, state: str, category: str, language_code: str) -> dict[str, Any]:
    """Gemini 2.5 Flash agent loop with Tavily + Firecrawl tools."""
    client = gemini_client(get_secret("GEMINI_API_KEY"))
    user_msg = (
        f"Farmer said: {transcript}\n"
        f"State: {state}\nCategory: {category}\nLanguage: {language_code}\n"
        "Search schemes, calculate subsidy, return JSON."
    )
    contents: list[types.Content] = [types.Content(role="user", parts=[types.Part(text=user_msg)])]

    for _ in range(8):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=_tools(),
                temperature=0.2,
            ),
        )
        candidate = response.candidates[0]
        parts = candidate.content.parts if candidate.content else []
        calls = [p.function_call for p in parts if p.function_call]

        if not calls:
            return _parse_json(response.text or "")

        contents.append(candidate.content)
        tool_parts = []
        for call in calls:
            tool_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=call.name,
                        response={"result": _run_tool(call.name, dict(call.args or {}), state)},
                    )
                )
            )
        contents.append(types.Content(role="user", parts=tool_parts))

    return _parse_json("Agent reached maximum search steps.")


# ---------------------------------------------------------------------------
# Formatting & UI helpers
# ---------------------------------------------------------------------------


def format_inr(amount: int | float) -> str:
    n = int(amount)
    if n <= 0:
        return "—"
    s = str(n)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return f"₹{','.join(parts + [last3])}"


def process_recording(audio_bytes: bytes, state: str, category: str, language_code: str) -> None:
    with st.spinner("🎙️ Listening… / सुन रहे हैं…"):
        transcript = transcribe(audio_bytes, language_code)
        st.session_state.transcript = transcript

    if not transcript:
        st.session_state.card_status = "error"
        return

    with st.spinner("🔍 Searching government schemes…"):
        result = run_agent(transcript, state, category, language_code)

    st.session_state.equipment = result.get("equipment_or_input", "")
    st.session_state.scheme_name = result.get("scheme_name", "—")
    st.session_state.subsidy_percent = int(result.get("subsidy_percent") or 0)
    st.session_state.max_claim_inr = int(result.get("max_claim_inr") or 0)
    st.session_state.missing_criteria = result.get("missing_criteria") or ""
    st.session_state.voice_response = result.get("voice_response", "")
    st.session_state.card_status = "warning" if st.session_state.missing_criteria else "success"

    with st.spinner("🔊 Generating voice response…"):
        try:
            st.session_state.tts_audio_bytes = text_to_speech(
                st.session_state.voice_response, language_code
            )
        except Exception as exc:
            st.session_state.tts_audio_bytes = None
            st.error(f"TTS failed: {exc}")


def render_metrics() -> None:
    status = st.session_state.card_status
    if status == "idle":
        st.markdown(
            '<p style="font-size:1.4rem;color:#558B2F;text-align:center;margin-top:3rem;">'
            "Speak into the microphone to search subsidies.<br/>"
            "Example: <em>\"Drip irrigation subsidy in Telangana\"</em>"
            "</p>",
            unsafe_allow_html=True,
        )
        return

    if st.session_state.scheme_name and st.session_state.scheme_name != "—":
        st.markdown(
            f'<div class="scheme-banner">📋 {st.session_state.scheme_name}</div>',
            unsafe_allow_html=True,
        )

    warn = status == "warning"
    pct = st.session_state.subsidy_percent
    claim = st.session_state.max_claim_inr

    st.markdown(
        f"""
<div class="metric-grid">
  <div class="metric-card{' warning' if warn else ''}">
    <div class="metric-label">Subsidy Percentage</div>
    <div class="metric-value">{pct}%</div>
  </div>
  <div class="metric-card{' warning' if warn else ''}">
    <div class="metric-label">Maximum Claimable Amount</div>
    <div class="metric-value highlight">{format_inr(claim)}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Equipment / Input</div>
    <div class="metric-value" style="font-size:1.6rem;">{st.session_state.equipment or '—'}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Eligible Scheme</div>
    <div class="metric-value" style="font-size:1.5rem;">{st.session_state.scheme_name or '—'}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.session_state.missing_criteria:
        st.markdown(
            f'<div class="missing-banner">⚠️ Missing: {st.session_state.missing_criteria}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<h1 class="main-header">🌾 Grameen Seva AI Hub</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="main-subtitle">Voice-First Government Subsidy Finder for Indian Farmers</p>',
    unsafe_allow_html=True,
)

missing = missing_secrets()
if missing:
    st.error(f"Missing API keys in secrets.toml: {', '.join(missing)}")
    st.stop()

# ---------------------------------------------------------------------------
# Two-column kiosk layout: 35% control | 65% dashboard
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([0.35, 0.65], gap="large")

with col_left:
    st.markdown('<p class="panel-title">🎙️ Voice Control Center</p>', unsafe_allow_html=True)

    with st.container(border=True):
        selected_state = st.selectbox("State", INDIAN_STATES, index=0)
        selected_category = st.selectbox("Farmer Category", FARMER_CATEGORIES)
        selected_lang_label = st.selectbox("Output Voice Language", list(LANGUAGE_OPTIONS.keys()))
        selected_language = LANGUAGE_OPTIONS[selected_lang_label]

    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    audio = st.audio_input(
        "Tap Mic & Speak / बोलने के लिए दबाएं",
        sample_rate=16000,
        key="kiosk_mic",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Optional QR for judges to open on mobile (uses deploy URL if set)
    deploy_url = get_secret("DEPLOY_URL") or "https://share.streamlit.io"
    qr = qrcode.make(deploy_url)
    buf = io.BytesIO()
    qr.save(buf, format="PNG")
    st.caption("Scan to open on your phone")
    st.image(buf.getvalue(), width=140)

with col_right:
    st.markdown('<p class="panel-title">📊 Subsidy Intelligence Dashboard</p>', unsafe_allow_html=True)

    if st.session_state.transcript:
        st.markdown(
            f'<div class="transcript-box">🗣️ <strong>You said:</strong> {st.session_state.transcript}</div>',
            unsafe_allow_html=True,
        )

    render_metrics()

    if st.session_state.tts_audio_bytes:
        st.markdown("##### 🔊 AI Voice Response")
        st.audio(st.session_state.tts_audio_bytes, format="audio/wav", autoplay=True)

        if st.button("🔊 Listen Again (फिर से सुनें)", use_container_width=True):
            st.session_state.replay_counter += 1
            st.audio(
                st.session_state.tts_audio_bytes,
                format="audio/wav",
                autoplay=True,
                key=f"replay_{st.session_state.replay_counter}",
            )

# Process new audio outside columns to avoid duplicate reruns
if audio is not None:
    audio_bytes = audio.getvalue()
    audio_hash = hash(audio_bytes)
    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash
        process_recording(audio_bytes, selected_state, selected_category, selected_language)
        st.rerun()
