import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Groq Chat", page_icon="⚡", layout="wide")

# ---------------- STYLING ---------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #FAF8F5 !important;
    font-family: 'DM Sans', sans-serif;
    color: #2C2A27;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Desktop: sidebar locked open ── */
@media (min-width: 768px) {
    [data-testid="stSidebar"] {
        transform: none !important;
        visibility: visible !important;
        display: block !important;
        min-width: 244px !important;
        width: 244px !important;
        left: 0 !important;
        position: relative !important;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }

    /* Hide mobile-only expander on desktop */
    .mobile-settings { display: none !important; }
}

/* ── Mobile: hide sidebar completely, show expander ── */
@media (max-width: 767px) {
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }

    .block-container {
        padding: 1rem 0.85rem 6rem 0.85rem !important;
        max-width: 100% !important;
    }
    .chat-header { padding: 0.75rem 0 1rem !important; }
    .chat-header h1 { font-size: 1.3rem !important; }
    .chat-header p  { font-size: 0.72rem !important; }
    .header-icon    { width: 30px !important; height: 30px !important; font-size: 0.9rem !important; line-height: 30px !important; }
    .bubble         { max-width: 88% !important; font-size: 0.875rem !important; }
    .avatar         { width: 24px !important; height: 24px !important; font-size: 0.6rem !important; }
}

/* ── Sidebar theming ── */
[data-testid="stSidebar"] {
    background-color: #F2EFE9 !important;
    border-right: 1px solid #E4DFD7 !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([data-baseweb]),
[data-testid="stSidebar"] div.stMarkdown {
    font-family: 'DM Sans', sans-serif !important;
    color: #2C2A27 !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Lora', serif !important;
    font-size: 1.05rem !important;
    color: #5C5142 !important;
    margin-bottom: 1rem;
}
[data-testid="stSidebar"] .stTooltipIcon,
[data-testid="stSidebar"] button[kind="icon"],
[data-testid="stSidebar"] [data-testid="tooltipHoverTarget"] { display: none !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] .stTextInput input {
    background-color: #FDFCFA !important;
    border: 1px solid #D9D3C9 !important;
    border-radius: 8px !important;
    color: #2C2A27 !important;
    font-size: 0.875rem !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child {
    background-color: #FDFCFA !important;
    border: 1px solid #D9D3C9 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] span { color: #2C2A27 !important; }
[data-baseweb="popover"] ul { background-color: #FDFCFA !important; }
[data-baseweb="popover"] li { color: #2C2A27 !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.875rem !important; }
[data-baseweb="popover"] li:hover { background-color: #EAE4DC !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #8B7D6B !important; border-color: #8B7D6B !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSliderTrack"] > div:nth-child(2) {
    background: linear-gradient(90deg, #F5A623, #F76B1C) !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important;
    border: 1px solid #C8BEB2 !important;
    border-radius: 8px !important;
    color: #6B5D51 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover { background-color: #EAE4DC !important; }
[data-testid="stSidebar"] label {
    font-size: 0.78rem !important; font-weight: 500 !important;
    letter-spacing: 0.05em !important; text-transform: uppercase !important;
    color: #9B8D7E !important;
}
[data-testid="stSidebar"] hr { border-color: #DDD6CC !important; margin: 1.25rem 0 !important; }

/* ── Mobile settings expander ── */
.mobile-settings [data-testid="stExpander"] {
    background-color: #F2EFE9 !important;
    border: 1px solid #E4DFD7 !important;
    border-radius: 12px !important;
    margin-bottom: 1.1rem !important;
}
.mobile-settings summary {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #5C5142 !important;
}
.mobile-settings label {
    font-size: 0.75rem !important; font-weight: 500 !important;
    letter-spacing: 0.04em !important; text-transform: uppercase !important;
    color: #9B8D7E !important;
}
.mobile-settings .stButton > button {
    background-color: transparent !important;
    border: 1px solid #C8BEB2 !important;
    border-radius: 8px !important;
    color: #6B5D51 !important;
    font-size: 0.82rem !important;
    width: 100%;
}

/* ── Main content ── */
[data-testid="stMain"] { background-color: #FAF8F5 !important; }
.block-container {
    max-width: 820px !important;
    padding: 2.5rem 2rem 6rem 2rem !important;
    margin: 0 auto !important;
}

/* ── Page Header ── */
.chat-header {
    text-align: center;
    padding: 1.5rem 0 2rem 0;
    border-bottom: 1px solid #E8E2D9;
    margin-bottom: 1.5rem;
}
.chat-header h1 {
    font-family: 'Lora', serif;
    font-size: 2rem; font-weight: 500;
    color: #2C2A27; letter-spacing: -0.02em;
    margin: 0 0 0.3rem 0;
}
.chat-header p {
    font-size: 0.82rem; color: #9B8D7E;
    letter-spacing: 0.08em; text-transform: uppercase; margin: 0;
}
.header-icon {
    display: inline-block;
    background: linear-gradient(135deg, #F5A623 0%, #F76B1C 100%);
    color: white; width: 38px; height: 38px;
    border-radius: 10px; font-size: 1.1rem;
    line-height: 38px; text-align: center;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 8px rgba(247,107,28,0.25);
}

/* ── Messages ── */
.message-row { display: flex; margin-bottom: 1.1rem; animation: fadeUp 0.25s ease; }
.message-row.user { justify-content: flex-end; }
.message-row.bot  { justify-content: flex-start; }
.avatar {
    width: 30px; height: 30px; border-radius: 50%;
    flex-shrink: 0; display: flex; align-items: center;
    justify-content: center; font-size: 0.75rem; font-weight: 600; margin-top: 2px;
}
.avatar-user { background: linear-gradient(135deg, #F5A623, #F76B1C); color: white; margin-left: 0.6rem; order: 2; }
.avatar-bot  { background: #EAE4DC; color: #6B5D51; margin-right: 0.6rem; }
.bubble {
    max-width: 72%; padding: 0.75rem 1rem;
    border-radius: 14px; font-size: 0.9rem;
    line-height: 1.65; word-break: break-word;
}
.bubble-user {
    background: linear-gradient(135deg, #F5A623 0%, #F76B1C 100%);
    color: #FFFFFF; border-bottom-right-radius: 4px;
    box-shadow: 0 2px 10px rgba(247,107,28,0.2);
}
.bubble-bot {
    background: #FFFFFF; color: #2C2A27;
    border: 1px solid #E8E2D9; border-bottom-left-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ── Empty state ── */
.empty-state { text-align: center; padding: 3rem 1rem; color: #B5A898; }
.empty-state .big-icon { font-size: 2.2rem; margin-bottom: 0.6rem; opacity: 0.45; }
.empty-state p { font-size: 0.88rem; font-family: 'Lora', serif; font-style: italic; margin: 0; }

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #FFFFFF !important; border: 1.5px solid #DDD6CC !important;
    border-radius: 14px !important; box-shadow: 0 2px 14px rgba(0,0,0,0.07) !important;
}
[data-testid="stChatInput"]:focus-within { border-color: #B5A080 !important; }
[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important; color: #2C2A27 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #B5A898 !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #F5A623, #F76B1C) !important;
    border-radius: 10px !important; border: none !important;
}

/* ── Misc ── */
[data-testid="stStatusWidget"] { display: none; }
[data-testid="stAlert"] {
    background-color: #FFF8F0 !important; border: 1px solid #F5C49B !important;
    border-radius: 10px !important; color: #7A4E2D !important; font-size: 0.85rem !important;
}
@keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #D9D3C9; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------- PROMPT ---------------- #
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, thoughtful assistant. Be concise but thorough."),
    ("user", "Question: {question}")
])

# ---------------- FUNCTION ---------------- #
def generate_response(question, api_key, model, temperature, max_tokens):
    llm = ChatGroq(
        groq_api_key=api_key, model_name=model,
        temperature=temperature, max_tokens=max_tokens
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({'question': question})

# ---------------- SESSION STATE ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Desktop sidebar ──────────────────────────────────────────────── #
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    api_key     = st.text_input("Groq API Key", type="password", placeholder="gsk_...", key="api_key")
    st.markdown("---")
    model       = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"], key="model")
    st.caption("8b = faster  ·  70b = smarter")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05, key="temperature")
    st.caption("Higher = more creative")
    max_tokens  = st.slider("Max Tokens", 50, 1000, 300, step=50, key="max_tokens")
    st.markdown("---")
    if st.button("🧹 Clear conversation", key="clear_btn"):
        st.session_state.messages = []
        st.rerun()
    st.markdown(
        "<p style='font-size:0.72rem;color:#B5A898;margin-top:1rem;text-align:center;'>"
        "Powered by Groq · LangChain</p>", unsafe_allow_html=True
    )

# ---------------- HEADER ---------------- #
st.markdown("""
<div class="chat-header">
    <div class="header-icon">⚡</div>
    <h1>Groq Chat</h1>
    <p>Fast inference · Open models</p>
</div>
""", unsafe_allow_html=True)

# ── Mobile settings expander (hidden on desktop via CSS) ─────────── #
st.markdown('<div class="mobile-settings">', unsafe_allow_html=True)
with st.expander("⚙️  Settings", expanded=False):
    mob_api_key     = st.text_input("Groq API Key", type="password", placeholder="gsk_...", key="mob_api_key")
    mob_model       = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"], key="mob_model")
    st.caption("8b = faster  ·  70b = smarter")
    mob_temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05, key="mob_temperature")
    st.caption("Higher = more creative")
    mob_max_tokens  = st.slider("Max Tokens", 50, 1000, 300, step=50, key="mob_max_tokens")
    if st.button("🧹 Clear conversation", key="mob_clear"):
        st.session_state.messages = []
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Prefer mobile inputs when provided
if st.session_state.get("mob_api_key"):
    api_key     = st.session_state["mob_api_key"]
    model       = st.session_state["mob_model"]
    temperature = st.session_state["mob_temperature"]
    max_tokens  = st.session_state["mob_max_tokens"]

# ---------------- CHAT DISPLAY ---------------- #
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="big-icon">💬</div>
        <p>Ask me anything — I'm quick.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="message-row user">
                <div class="bubble bubble-user">{msg['content']}</div>
                <div class="avatar avatar-user">You</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-row bot">
                <div class="avatar avatar-bot">⚡</div>
                <div class="bubble bubble-bot">{msg['content']}</div>
            </div>""", unsafe_allow_html=True)

# ---------------- INPUT ---------------- #
user_input = st.chat_input("Type a message…")

# ---------------- RESPONSE ---------------- #
if user_input:
    if not api_key:
        st.warning("⚠️  Please enter your Groq API key in Settings to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner(""):
        try:
            response = generate_response(user_input, api_key, model, temperature, max_tokens)
        except Exception as e:
            response = f"⚠️ {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()