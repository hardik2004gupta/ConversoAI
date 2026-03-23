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

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #FAF8F5 !important;
    font-family: 'DM Sans', sans-serif;
    color: #2C2A27;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ════════════════════════════════════════
   DESKTOP  (≥ 768px) — sidebar locked open
   ════════════════════════════════════════ */
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
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
    #mob-fab, #mob-backdrop { display: none !important; }
}

/* ════════════════════════════════════════
   MOBILE  (< 768px) — drawer overlay
   ════════════════════════════════════════ */
@media (max-width: 767px) {

    /* Hide native toggle buttons */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    /* Sidebar becomes a fixed full-height drawer, off-screen by default */
    [data-testid="stSidebar"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        height: 100dvh !important;
        width: 88vw !important;
        max-width: 320px !important;
        z-index: 1100 !important;
        transform: translateX(-110%) !important;
        transition: transform 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 4px 0 32px rgba(0,0,0,0.13) !important;
        overflow-y: auto !important;
        padding-top: 1rem !important;
    }
    /* Open state — JS adds this class to <body> */
    body.sidebar-open [data-testid="stSidebar"] {
        transform: translateX(0) !important;
    }

    /* Dim backdrop */
    #mob-backdrop {
        display: block;
        position: fixed;
        inset: 0;
        background: rgba(44, 42, 39, 0.45);
        z-index: 1099;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.28s ease;
    }
    body.sidebar-open #mob-backdrop {
        opacity: 1;
        pointer-events: all;
    }

    /* Give main content full width */
    [data-testid="stAppViewContainer"] > section:last-child {
        margin-left: 0 !important;
        width: 100% !important;
    }

    /* Tighter main padding on mobile */
    .block-container {
        padding: 1rem 1rem 7rem 1rem !important;
    }

    /* Smaller header on mobile */
    .chat-header { padding: 1rem 0 1.25rem 0 !important; }
    .chat-header h1 { font-size: 1.4rem !important; }

    /* Bubbles fill more width on narrow screens */
    .bubble { max-width: 88% !important; }

    /* Larger tap targets for avatars */
    .avatar { width: 26px !important; height: 26px !important; font-size: 0.65rem !important; }

    /* Chat input sits above mobile keyboard */
    [data-testid="stBottom"] {
        padding-bottom: env(safe-area-inset-bottom, 0.5rem) !important;
    }
}

/* ── Floating Action Button (mobile only) ── */
#mob-fab {
    position: fixed;
    bottom: 5.5rem;
    right: 1.1rem;
    z-index: 1200;
    width: 46px;
    height: 46px;
    border-radius: 14px;
    background: #FFFFFF;
    border: 1px solid #E0D9D0;
    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.15s, box-shadow 0.15s;
    -webkit-tap-highlight-color: transparent;
}
#mob-fab:active { transform: scale(0.93); box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
#mob-fab svg {
    width: 18px; height: 18px;
    stroke: #6B5D51; stroke-width: 2;
    stroke-linecap: round; fill: none;
}

/* ── Sidebar theming (shared desktop + mobile) ── */
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
    letter-spacing: 0.01em;
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
    padding: 0.45rem 0.75rem !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: #B5A898 !important;
    box-shadow: 0 0 0 3px rgba(181,168,152,0.18) !important;
    outline: none !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child {
    background-color: #FDFCFA !important;
    border: 1px solid #D9D3C9 !important;
    border-radius: 8px !important;
    font-size: 0.875rem !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #2C2A27 !important;
}
[data-baseweb="popover"] ul { background-color: #FDFCFA !important; }
[data-baseweb="popover"] li {
    color: #2C2A27 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
}
[data-baseweb="popover"] li:hover { background-color: #EAE4DC !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #8B7D6B !important;
    border-color: #8B7D6B !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSliderTrack"] > div:first-child {
    background-color: #D9D3C9 !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSliderTrack"] > div:nth-child(2) {
    background: linear-gradient(90deg, #F5A623, #F76B1C) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    color: #6B5D51 !important;
    font-size: 0.78rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: #9B8D7E !important;
    font-size: 0.72rem !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important;
    border: 1px solid #C8BEB2 !important;
    border-radius: 8px !important;
    color: #6B5D51 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    transition: background 0.2s, border-color 0.2s !important;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #EAE4DC !important;
    border-color: #A89880 !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    font-size: 0.72rem !important;
    color: #B5A898 !important;
    margin-top: -0.4rem !important;
    margin-bottom: 0.6rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #9B8D7E !important;
    margin-bottom: 0.3rem !important;
}
[data-testid="stSidebar"] hr {
    border-color: #DDD6CC !important;
    margin: 1.25rem 0 !important;
}

/* ── Main Content Area ── */
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
    font-size: 2rem;
    font-weight: 500;
    color: #2C2A27;
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem 0;
}
.chat-header p {
    font-size: 0.82rem;
    color: #9B8D7E;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0;
}
.header-icon {
    display: inline-block;
    background: linear-gradient(135deg, #F5A623 0%, #F76B1C 100%);
    color: white;
    width: 38px; height: 38px;
    border-radius: 10px;
    font-size: 1.1rem;
    line-height: 38px;
    text-align: center;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 8px rgba(247,107,28,0.25);
}

/* ── Chat Messages ── */
.message-row {
    display: flex;
    margin-bottom: 1.1rem;
    animation: fadeUp 0.25s ease;
}
.message-row.user { justify-content: flex-end; }
.message-row.bot  { justify-content: flex-start; }
.avatar {
    width: 30px; height: 30px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 600;
    margin-top: 2px;
}
.avatar-user {
    background: linear-gradient(135deg, #F5A623, #F76B1C);
    color: white; margin-left: 0.6rem; order: 2;
}
.avatar-bot { background: #EAE4DC; color: #6B5D51; margin-right: 0.6rem; }
.bubble {
    max-width: 72%;
    padding: 0.75rem 1rem;
    border-radius: 14px;
    font-size: 0.9rem;
    line-height: 1.65;
    word-break: break-word;
}
.bubble-user {
    background: linear-gradient(135deg, #F5A623 0%, #F76B1C 100%);
    color: #FFFFFF;
    border-bottom-right-radius: 4px;
    box-shadow: 0 2px 10px rgba(247,107,28,0.2);
}
.bubble-bot {
    background: #FFFFFF; color: #2C2A27;
    border: 1px solid #E8E2D9;
    border-bottom-left-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ── Empty State ── */
.empty-state { text-align: center; padding: 3.5rem 1rem; color: #B5A898; }
.empty-state .big-icon { font-size: 2.5rem; margin-bottom: 0.75rem; opacity: 0.5; }
.empty-state p {
    font-size: 0.88rem; font-family: 'Lora', serif;
    font-style: italic; margin: 0;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1.5px solid #DDD6CC !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 14px rgba(0,0,0,0.07) !important;
    padding: 0.2rem 0.5rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #B5A080 !important;
    box-shadow: 0 2px 18px rgba(0,0,0,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    color: #2C2A27 !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #B5A898 !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #F5A623, #F76B1C) !important;
    border-radius: 10px !important; border: none !important;
    color: white !important; transition: opacity 0.15s !important;
}
[data-testid="stChatInput"] button:hover { opacity: 0.88 !important; }

/* ── Spinner / Status ── */
[data-testid="stStatusWidget"] { display: none; }
.stSpinner > div { border-top-color: #F5A623 !important; }

/* ── Warning / Error ── */
[data-testid="stAlert"] {
    background-color: #FFF8F0 !important;
    border: 1px solid #F5C49B !important;
    border-radius: 10px !important;
    color: #7A4E2D !important;
    font-size: 0.85rem !important;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D9D3C9; border-radius: 10px; }
</style>

<!-- Mobile backdrop + FAB -->
<div id="mob-backdrop"></div>
<button id="mob-fab" aria-label="Open settings">
    <svg viewBox="0 0 22 22" xmlns="http://www.w3.org/2000/svg">
        <circle cx="11" cy="11" r="3"/>
        <path d="M11 2v2M11 18v2M2 11h2M18 11h2M4.93 4.93l1.41 1.41M15.66 15.66l1.41 1.41M4.93 17.07l1.41-1.41M15.66 6.34l1.41-1.41"/>
    </svg>
</button>

<script>
(function () {
    function isMobile() { return window.innerWidth < 768; }

    var fab      = document.getElementById('mob-fab');
    var backdrop = document.getElementById('mob-backdrop');

    function openDrawer()  { document.body.classList.add('sidebar-open'); }
    function closeDrawer() { document.body.classList.remove('sidebar-open'); }

    if (fab)      fab.addEventListener('click', function() {
        document.body.classList.contains('sidebar-open') ? closeDrawer() : openDrawer();
    });
    if (backdrop) backdrop.addEventListener('click', closeDrawer);

    // Close on resize to desktop
    window.addEventListener('resize', function() {
        if (!isMobile()) closeDrawer();
    });
})();
</script>
""", unsafe_allow_html=True)

# ---------------- PROMPT ---------------- #
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, thoughtful assistant. Be concise but thorough."),
        ("user", "Question: {question}")
    ]
)

# ---------------- FUNCTION ---------------- #
def generate_response(question, api_key, model, temperature, max_tokens):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({'question': question})


# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.markdown("---")

    model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"]
    )
    st.caption("8b = faster  ·  70b = smarter")

    st.markdown("")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05)
    st.caption("Higher = more creative")
    max_tokens = st.slider("Max Tokens", 50, 1000, 300, step=50)

    st.markdown("---")

    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<p style='font-size:0.72rem;color:#B5A898;margin-top:1.5rem;text-align:center;'>"
        "Powered by Groq · LangChain</p>",
        unsafe_allow_html=True
    )


# ---------------- CHAT STATE ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- HEADER ---------------- #
st.markdown("""
<div class="chat-header">
    <div class="header-icon">⚡</div>
    <h1>Groq Chat</h1>
    <p>Fast inference · Open models</p>
</div>
""", unsafe_allow_html=True)

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
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-row bot">
                <div class="avatar avatar-bot">⚡</div>
                <div class="bubble bubble-bot">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------- INPUT ---------------- #
user_input = st.chat_input("Type a message…")

# ---------------- RESPONSE ---------------- #
if user_input:
    if not api_key:
        st.warning("⚠️  Please enter your Groq API key in the sidebar to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner(""):
        try:
            response = generate_response(
                user_input, api_key, model, temperature, max_tokens
            )
        except Exception as e:
            response = f"⚠️ {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()