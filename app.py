import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Groq Chat", page_icon="⚡", layout="centered")

# ---------------- STYLING ---------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@400;500;600&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body {
    background-color: #FAF8F5 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #1A1816 !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background-color: #FAF8F5 !important;
}

/* ── Hide Streamlit chrome & sidebar entirely ── */
#MainMenu, footer, header,
[data-testid="stSidebar"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}

/* ── Layout ── */
.block-container {
    max-width: 680px !important;
    padding: 1.75rem 1.25rem 6rem 1.25rem !important;
    margin: 0 auto !important;
}

/* ── Page Header ── */
.chat-header {
    text-align: center;
    padding: 1.25rem 0 1.5rem 0;
    border-bottom: 1px solid #E8E2D9;
    margin-bottom: 1.25rem;
}
.chat-header h1 {
    font-family: 'Lora', serif !important;
    font-size: 1.85rem;
    font-weight: 500;
    color: #1A1816 !important;
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem 0;
}
.chat-header p {
    font-size: 0.78rem;
    color: #7A6E65 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0;
}
.header-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #F5A623 0%, #F76B1C 100%);
    color: white;
    width: 36px; height: 36px;
    border-radius: 10px;
    font-size: 1rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 8px rgba(247,107,28,0.28);
}

/* ── Settings expander ── */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #E0D9D0 !important;
    border-radius: 14px !important;
    margin-bottom: 1.25rem !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
    background-color: #FFFFFF !important;
    color: #1A1816 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1rem !important;
}
/* Expander inner content background */
[data-testid="stExpander"] > div:last-child {
    background-color: #FDFCFA !important;
    padding: 0.5rem 1rem 1rem 1rem !important;
    border-top: 1px solid #EDE7DE !important;
}

/* ── ALL labels — force dark, visible ── */
label,
.stTextInput label,
.stSelectbox label,
.stSlider label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-baseweb="form-control-label"],
p.st-emotion-cache-label {
    color: #1A1816 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* ── Text inputs ── */
input[type="text"],
input[type="password"],
input[type="number"],
.stTextInput input {
    background-color: #FFFFFF !important;
    border: 1.5px solid #D4CCC3 !important;
    border-radius: 9px !important;
    color: #1A1816 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 0.75rem !important;
}
input[type="text"]:focus,
input[type="password"]:focus,
.stTextInput input:focus {
    border-color: #C8A882 !important;
    box-shadow: 0 0 0 3px rgba(200,168,130,0.2) !important;
    outline: none !important;
}
input[type="text"]::placeholder,
input[type="password"]::placeholder {
    color: #A89880 !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] > div:first-child {
    background-color: #FFFFFF !important;
    border: 1.5px solid #D4CCC3 !important;
    border-radius: 9px !important;
    color: #1A1816 !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: #1A1816 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-baseweb="popover"] ul { background-color: #FFFFFF !important; }
[data-baseweb="popover"] li {
    color: #1A1816 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-baseweb="popover"] li:hover { background-color: #F2EDE6 !important; }

/* ── Sliders ── */
[data-testid="stSlider"] [role="slider"] {
    background-color: #8B7D6B !important;
    border-color: #8B7D6B !important;
}
[data-testid="stSlider"] [data-testid="stSliderTrack"] > div:nth-child(2) {
    background: linear-gradient(90deg, #F5A623, #F76B1C) !important;
}
[data-testid="stSliderThumbValue"],
[data-testid="stSlider"] p {
    color: #1A1816 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
[data-testid="stTickBarMin"],
[data-testid="stTickBarMax"] {
    color: #7A6E65 !important;
    font-size: 0.72rem !important;
}

/* ── Caption text ── */
[data-testid="stCaptionContainer"] p,
.stCaption, small {
    color: #7A6E65 !important;
    font-size: 0.74rem !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: transparent !important;
    border: 1.5px solid #C8BEB2 !important;
    border-radius: 9px !important;
    color: #3D3530 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    transition: background 0.18s, border-color 0.18s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background-color: #F0EBE3 !important;
    border-color: #A89880 !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #E8E2D9 !important;
    margin: 1rem 0 !important;
}

/* ── Messages ── */
.message-row {
    display: flex;
    margin-bottom: 1rem;
    animation: fadeUp 0.22s ease;
}
.message-row.user { justify-content: flex-end; }
.message-row.bot  { justify-content: flex-start; }

.avatar {
    width: 28px; height: 28px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700;
    margin-top: 3px;
}
.avatar-user {
    background: linear-gradient(135deg, #F5A623, #F76B1C);
    color: #FFFFFF;
    margin-left: 0.55rem;
    order: 2;
}
.avatar-bot {
    background: #EAE4DC;
    color: #5C5044;
    margin-right: 0.55rem;
}

.bubble {
    max-width: 78%;
    padding: 0.7rem 0.95rem;
    border-radius: 14px;
    font-size: 0.9rem;
    line-height: 1.6;
    word-break: break-word;
    font-family: 'DM Sans', sans-serif !important;
}
.bubble-user {
    background: linear-gradient(135deg, #F5A623 0%, #F76B1C 100%);
    color: #FFFFFF !important;
    border-bottom-right-radius: 4px;
    box-shadow: 0 2px 10px rgba(247,107,28,0.22);
}
.bubble-bot {
    background: #FFFFFF;
    color: #1A1816 !important;
    border: 1.5px solid #E8E2D9;
    border-bottom-left-radius: 4px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.05);
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #A89880 !important;
}
.empty-state .big-icon { font-size: 2rem; margin-bottom: 0.5rem; opacity: 0.5; }
.empty-state p {
    font-size: 0.9rem;
    font-family: 'Lora', serif !important;
    font-style: italic;
    color: #A89880 !important;
    margin: 0;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1.5px solid #D4CCC3 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 14px rgba(0,0,0,0.07) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #C8A882 !important;
    box-shadow: 0 2px 18px rgba(0,0,0,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    color: #1A1816 !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #A89880 !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #F5A623, #F76B1C) !important;
    border-radius: 10px !important;
    border: none !important;
}

/* ── Warning / alert ── */
[data-testid="stStatusWidget"] { display: none; }
[data-testid="stAlert"] {
    background-color: #FFF8F0 !important;
    border: 1px solid #F5C49B !important;
    border-radius: 10px !important;
    color: #6B3A1A !important;
    font-size: 0.87rem !important;
}
[data-testid="stAlert"] p { color: #6B3A1A !important; }

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(7px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #D4CCC3; border-radius: 8px; }

/* ── Mobile tweaks ── */
@media (max-width: 600px) {
    .block-container { padding: 1rem 0.75rem 6rem 0.75rem !important; }
    .chat-header h1  { font-size: 1.45rem !important; }
    .bubble          { max-width: 90% !important; font-size: 0.875rem !important; }
}
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

# ---------------- HEADER ---------------- #
st.markdown("""
<div class="chat-header">
    <div class="header-icon">⚡</div>
    <h1>Groq Chat</h1>
    <p>Fast inference · Open models</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SETTINGS EXPANDER ---------------- #
with st.expander("⚙️  Settings", expanded=False):
    api_key = st.text_input("Groq API Key", type="password",
                            placeholder="gsk_...", key="api_key")
    st.divider()
    model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"],
        key="model"
    )
    st.caption("8b = faster  ·  70b = smarter")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05, key="temperature")
    st.caption("Higher = more creative")
    max_tokens = st.slider("Max Tokens", 50, 1000, 300, step=50, key="max_tokens")
    st.divider()
    if st.button("🧹 Clear conversation", key="clear_btn"):
        st.session_state.messages = []
        st.rerun()
    st.markdown(
        "<p style='font-size:0.72rem; color:#A89880; margin-top:0.75rem; text-align:center;'>"
        "Powered by Groq · LangChain</p>",
        unsafe_allow_html=True
    )

# Read values from session state
api_key     = st.session_state.get("api_key", "")
model       = st.session_state.get("model", "llama-3.1-8b-instant")
temperature = st.session_state.get("temperature", 0.7)
max_tokens  = st.session_state.get("max_tokens", 300)

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
        st.warning("⚠️  Please enter your Groq API key in Settings above.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner(""):
        try:
            response = generate_response(user_input, api_key, model, temperature, max_tokens)
        except Exception as e:
            response = f"⚠️ {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()