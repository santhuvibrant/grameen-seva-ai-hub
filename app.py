import streamlit as st
import time
import logging
from google import genai
from google.genai.errors import APIError

# Configure logging
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------
# GEMINI CONFIGURATION (google-genai SDK v1.25+)
# ---------------------------------------------------------
GEMINI_KEY = st.secrets.get("gemini_api_key") or st.secrets.get("GEMINI_API_KEY")
ai_client = None
if GEMINI_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
else:
    st.error("Configuration Error: Gemini API Key missing from Streamlit secrets.")

# ---------------------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------------------
from models.conversation import ConversationState
from agents.conversation import process_conversation, AgentResult
from services.recorder import autonomous_recorder
from services.sarvam import speech_to_text, text_to_speech
from services.research import tavily_search, firecrawl_scrape

# ---------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Grameen Seva AI Hub",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Kiosk UI
st.markdown("""
    <style>
    /* Hide default Streamlit Chrome and raw Audio elements */
    #MainMenu, footer, header { visibility: hidden !important; }
    audio { display: none !important; visibility: hidden !important; height: 0px !important; }
    
    .stApp {
        background-color: #F8FBF8;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    .header-container {
        text-align: center;
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    .main-title {
        color: #1B5E20;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        color: #388E3C;
        font-size: 1.15rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .lang-badge {
        display: inline-block;
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
        border: 1px solid #A5D6A7;
        margin-bottom: 1.5rem;
    }
    .chat-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        border: 1px solid #E0E0E0;
    }
    .user-label {
        color: #2E7D32;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 4px;
    }
    .user-text {
        color: #1C2A1E;
        font-size: 1.1rem;
        line-height: 1.5;
        margin-bottom: 16px;
    }
    .ai-label {
        color: #1B5E20;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 4px;
    }
    .ai-text {
        color: #263238;
        font-size: 1.1rem;
        line-height: 1.6;
        border-left: 4px solid #4CAF50;
        padding-left: 12px;
        background-color: #FAFAFA;
        padding-top: 8px;
        padding-bottom: 8px;
        border-radius: 0 8px 8px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
def init_state():
    if "state" not in st.session_state:
        st.session_state.state = ConversationState(
            language=None,
            history=[],
            district=None,
            state=None,
            land_size=None,
            farmer_category=None,
            equipment_need=None,
            documents=[],
            eligibility_status=None,
            conversation_complete=False
        )
    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None
    if "tts_audio_b64" not in st.session_state:
        st.session_state.tts_audio_b64 = None

# ---------------------------------------------------------
# GOV SEARCH PIPELINE WITH GEMINI SUMMARIZATION
# ---------------------------------------------------------
def run_gov_search_pipeline(state: ConversationState) -> str:
    """Executes Tavily search restricted to official government domains, scrapes via Firecrawl, and summarizes."""
    district = state.district or ""
    st_name = state.state or ""
    category = state.farmer_category or ""
    equipment = state.equipment_need or ""
    
    query = f"government scheme subsidy {equipment} {category} {district} {st_name} site:myscheme.gov.in OR site:gov.in"
    
    try:
        search_results = tavily_search(query)
        if search_results and "results" in search_results and search_results["results"]:
            official_url = search_results["results"][0].get("url")
            if official_url and ("gov.in" in official_url):
                page_text = firecrawl_scrape(official_url)
                if page_text and ai_client:
                    prompt = f"""
                    You are assisting an Indian farmer. Summarize the government subsidy scheme details clearly and concisely.
                    Rely strictly on the provided web content. Do NOT invent eligibility or amounts.
                    MUST reply entirely in the farmer's detected language (Language code: {state.language}).
                    
                    Scraped Web Content:
                    {page_text[:4000]}
                    """
                    for attempt in range(3):
                        try:
                            res = ai_client.models.generate_content(
                                model="gemini-1.5-flash",
                                contents=prompt
                            )
                            if res and res.text:
                                return res.text
                        except APIError as e:
                            logging.warning(f"Gemini API attempt {attempt+1} retry: {e}")
                            time.sleep(1.5 * (attempt + 1))
    except Exception as e:
        logging.error(f"Search pipeline error: {e}")
        
    fallback_map = {
        "te-IN": "మీ పరిధిలోని ప్రభుత్వ సబ్సిడీ వివరాల సమాచారం కోసం అధికారిక వెబ్‌సైట్ myscheme.gov.in చూడవచ్చు.",
        "hi-IN": "आपकी सरकारी सब्सिडी योजना की जानकारी के लिए myscheme.gov.in पर देखें।",
        "ta-IN": "உங்கள் அரசு மானிய திட்ட தகவல்களுக்கு myscheme.gov.in வலைத்தளத்தை பார்க்கவும்."
    }
    return fallback_map.get(state.language, "For official government scheme information, please visit myscheme.gov.in.")

# ---------------------------------------------------------
# MAIN KIOSK APPLICATION
# ---------------------------------------------------------
def main():
    init_state()
    
    # 1. Single Title & Subtitle Header
    st.markdown("""
        <div class="header-container">
            <div class="main-title">🌾 Grameen Seva AI Hub</div>
            <div class="sub-title">Voice-First Government Subsidy Finder for Indian Farmers</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. Dynamic Language Indicator
    if st.session_state.state.language:
        lang_display = {
            "te-IN": "తెలుగు (Telugu)",
            "hi-IN": "हिंदी (Hindi)",
            "ta-IN": "தமிழ் (Tamil)",
            "kn-IN": "ಕನ್ನಡ (Kannada)",
            "ml-IN": "മലയാളം (Malayalam)",
            "mr-IN": "మరాठी (Marathi)",
            "bn-IN": "বাংলা (Bengali)",
            "gu-IN": "ગુજરાતી (Gujarati)",
            "pa-IN": "ਪੰਜਾਬੀ (Punjabi)",
            "en-IN": "English"
        }.get(st.session_state.state.language, st.session_state.state.language)
        
        st.markdown(f'<div style="text-align:center;"><span class="lang-badge">🌐 Detected Language: {lang_display}</span></div>', unsafe_allow_html=True)

    # 3. Conversation History Card
    if st.session_state.state.history:
        st.markdown('<div class="chat-card">', unsafe_allow_html=True)
        for msg in st.session_state.state.history:
            if msg["role"] == "user":
                st.markdown('<div class="user-label">🧑‍🌾 Farmer:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="user-text">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ai-label">🤖 Grameen AI:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ai-text">{msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Microphone Kiosk Interaction
    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        audio_data = autonomous_recorder()

    # 5. Speech & Agent Pipeline Processing
    if audio_data:
        curr_hash = hash(audio_data)
        if curr_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = curr_hash
            
            with st.spinner("Processing speech..."):
                # STT Processing via Sarvam
                stt_res = speech_to_text(audio_data) or {}
                transcript = stt_res.get("transcript")
                detected_lang = stt_res.get("language_code")
                
                if transcript:
                    # Automatically set detected language state
                    if detected_lang and not st.session_state.state.language:
                        st.session_state.state.language = detected_lang
                    elif not st.session_state.state.language:
                        st.session_state.state.language = "hi-IN"
                        
                    st.session_state.state.history.append({"role": "user", "content": transcript})
                    
                    # Agent Conversation Step
                    try:
                        agent_res: AgentResult = process_conversation(
                            st.session_state.state,
                            transcript
                        )
                        
                        if agent_res and hasattr(agent_res, "response_text"):
                            reply_text = agent_res.response_text
                            st.session_state.state = agent_res.updated_state
                        elif isinstance(agent_res, dict):
                            reply_text = agent_res.get("response_text", "")
                        else:
                            reply_text = "దయచేసి మీ ప్రశ్నను మరొకసారి తెలపండి."
                    except Exception as err:
                        logging.error(f"Error in conversation processing: {err}")
                        reply_text = "దయచేసి మీ ప్రశ్నను మరొకసారి తెలపండి."
                    
                    # Search Pipeline Trigger (when profile information is complete)
                    if getattr(st.session_state.state, "conversation_complete", False):
                        search_summary = run_gov_search_pipeline(st.session_state.state)
                        if search_summary:
                            reply_text = search_summary
                            
                    st.session_state.state.history.append({"role": "assistant", "content": reply_text})
                    
                    # TTS Voice Response via Sarvam
                    tts_b64 = text_to_speech(reply_text, st.session_state.state.language)
                    st.session_state.tts_audio_b64 = tts_b64
                    
                    st.rerun()

    # 6. Invisible Background Audio Autoplay
    if st.session_state.get("tts_audio_b64"):
        b64_str = st.session_state.tts_audio_b64
        st.markdown(
            f'''
            <audio autoplay style="display:none !important; visibility:hidden !important;">
                <source src="data:audio/wav;base64,{b64_str}" type="audio/wav">
            </audio>
            ''',
            unsafe_allow_html=True
        )
        st.session_state.tts_audio_b64 = None

if __name__ == "__main__":
    main()
