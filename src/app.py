import streamlit as st
import time
from rag import RAGAssistant, load_documents 

@st.cache_resource
def initialize_rag_assistant():
    """Initializes the RAG Assistant and loads documents into the VectorDB."""
    try:
        st.info("Initializing RAG Assistant and loading documents (this may take a moment)...")
        docs = load_documents() 
        assistant = RAGAssistant()
        assistant.add_documents(docs)
        st.success(f"Loaded {len(docs)} documents successfully. Start chatting!")
        return assistant
    except Exception as e:
        st.error(f"Failed to initialize assistant or load documents. Check API keys and file paths. Error: {e}")
        st.stop()
        
# Initialize the assistant and data loading
rag_assistant = initialize_rag_assistant()


st.set_page_config(
    page_title="RAG-GenAI Terminal",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Global Background and Typography */
    .main {
        background: #0a0f18; /* Deep Midnight Blue/Charcoal */
        color: #e0e0e0; /* Light Gray Text */
        font-family: 'Consolas', 'Courier New', monospace; 
    }
    .stApp, .stTextInput > div > div > input, .stTextArea > label {
        color: #00FFFF !important; /* Neon Cyan accent for input text */
    }
    
    /* Message Container (Scrollable) */
    .stChatMessage {
        background: transparent !important;
        border-radius: 10px;
        padding: 0;
        margin-bottom: 15px;
        color: #e0e0e0;
    }

    /* User Message Bubble */
    .user-msg {
        background-color: #1a2a47; /* Darker blue background */
        color: #ffffff;
        padding: 12px 20px;
        border-radius: 15px 15px 0 15px;
        margin-left: 30%; /* Align right (ish) */
        text-align: left;
        border: 1px solid #00FFFF22; /* Subtle cyan border */
        box-shadow: 0 0 5px #00FFFF20;
        animation: slideInRight 0.3s ease-out;
    }

    /* Bot Message Bubble (Glowing Effect) */
    .bot-msg {
        background-color: #121c2c; /* Deep charcoal/blue background */
        color: #00FFFF; /* Neon Cyan text */
        padding: 12px 20px;
        border-radius: 15px 15px 15px 0;
        margin-right: 30%; /* Align left (ish) */
        text-align: left;
        border: 1px solid #00FFFF; /* Electric border */
        box-shadow: 0 0 15px #00FFFF50, 0 0 5px #00FFFF; /* Dashing glow */
        animation: slideInLeft 0.3s ease-out;
        transition: all 0.2s;
    }

    /* Input Field & Form Container Styling */
    .stTextInput label, .stForm {
        color: #00FFFF;
    }
    .stTextInput > div > div > input {
        background-color: #0a0f18; /* Same as background */
        border: 2px solid #00FFFF; /* Electric cyan border */
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 0 10px #00FFFF40;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00FF00; /* Green focus border */
        box-shadow: 0 0 15px #00FF0080;
    }

    /* Send Button Styling (Pulsating Effect) */
    .stButton>button {
        background-color: #00FFFF;
        color: #0a0f18;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        border: none;
        transition: transform 0.2s, background-color 0.2s;
        animation: pulse 1.5s infinite; /* Dashing Pulse animation */
    }
    .stButton>button:hover {
        background-color: #00FF00; /* Hover color change */
        transform: scale(1.05);
    }
    
    /* Keyframe Animations */
    @keyframes slideInLeft {
        0% { opacity: 0; transform: translateX(-50px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideInRight {
        0% { opacity: 0; transform: translateX(50px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 255, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0, 255, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 255, 0); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Chatbot Interface
# ---------------------------------------------------------

st.markdown("<h1 style='text-align: center; color: #00FFFF; text-shadow: 0 0 10px #00FFFF;'>RAG Terminal Interface</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Query your knowledge base. Use 'summary' for chat history.</p>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f'<div class="user-msg">{chat["msg"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{chat["msg"]}</div>', unsafe_allow_html=True)

# --- Chat Input Form ---
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input("Enter Command:", "", label_visibility="collapsed")
    with col2:
        submitted = st.form_submit_button("SEND")

# --- Submission Logic ---
if submitted and user_input.strip() != "":
    st.session_state.chat_history.append({"role": "user", "msg": user_input})
    st.rerun() # Use rerun to immediately show the new user message

# Logic executed after rerun
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    user_query = st.session_state.chat_history[-1]["msg"]
    
    with st.empty():
        # Display the user message once more to prepare for the bot response immediately after
        st.markdown(f'<div class="user-msg">{user_query}</div>', unsafe_allow_html=True)
        # Show a "loading" message in the style of the bot
        placeholder = st.markdown('<div class="bot-msg">...Awaiting Response...</div>', unsafe_allow_html=True)

    # 3. Generate response
    try:
        time.sleep(0.5) 
        bot_response = rag_assistant.invoke(user_query)

        # 4. Display the final response
        placeholder.markdown(f'<div class="bot-msg">{bot_response}</div>', unsafe_allow_html=True)
        
        # 5. Save final response
        st.session_state.chat_history.append({"role": "bot", "msg": bot_response})
    
    except Exception as e:
        error_msg = f"An error occurred: {e}"
        placeholder.markdown(f'<div class="bot-msg" style="color: #FF0000; border-color: #FF0000;">{error_msg}</div>', unsafe_allow_html=True)
        # Remove the user message if it failed, or save the error
        st.session_state.chat_history.pop() # Remove user query