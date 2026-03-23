import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Groq Chatbot", page_icon="⚡", layout="wide")

# ---------------- STYLING ---------------- #
st.markdown("""
<style>
.chat-container {
    max-width: 800px;
    margin: auto;
}
.user-msg {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: right;
}
.bot-msg {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PROMPT ---------------- #
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
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
st.sidebar.title("⚙️ Settings")

api_key = st.sidebar.text_input("Groq API Key", type="password")

model = st.sidebar.selectbox(
    "Model",
    ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"]
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 1000, 200)

# Clear chat
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []

# ---------------- CHAT STATE ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- TITLE ---------------- #
st.title("⚡ Groq Chatbot")
st.caption("Fast • Free • Hackathon Ready")

# ---------------- CHAT DISPLAY ---------------- #
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)

# ---------------- INPUT ---------------- #
user_input = st.chat_input("Type your message...")

# ---------------- RESPONSE ---------------- #
if user_input:
    if not api_key:
        st.warning("⚠️ Please enter your Groq API key")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking... ⚡"):
        try:
            response = generate_response(
                user_input,
                api_key,
                model,
                temperature,
                max_tokens
            )
        except Exception as e:
            response = f"Error: {str(e)}"

    # Add bot response
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()