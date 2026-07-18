"""
main.py — Streamlit entrypoint.

Handles: login/register/logout, listing + switching past conversations,
starting new chats, displaying message history, and running the
LangChain+Chroma chain on each new user message.
"""

import streamlit as st
import extra_streamlit_components as stx

import db
import auth
import chat

st.set_page_config(page_title="My Chatbot", page_icon="💬", layout="wide")

db.init_db()


# ---------- cookie-based "remember me" so a page refresh doesn't log you out ----------

@st.cache_resource
def get_cookie_manager():
    return stx.CookieManager()

cookie_manager = get_cookie_manager()


def _restore_session_from_cookie():
    if "user_id" in st.session_state:
        return
    cookie_user_id = cookie_manager.get("user_id")
    cookie_username = cookie_manager.get("username")
    if cookie_user_id and cookie_username:
        st.session_state.user_id = int(cookie_user_id)
        st.session_state.username = cookie_username


def _start_session(user_id: int, username: str):
    st.session_state.user_id = user_id
    st.session_state.username = username
    cookie_manager.set("user_id", str(user_id))
    cookie_manager.set("username", username)


def _end_session():
    for key in ("user_id", "username", "conversation_id", "messages"):
        st.session_state.pop(key, None)
    cookie_manager.delete("user_id")
    cookie_manager.delete("username")


_restore_session_from_cookie()


# ---------- login / register screen ----------

def render_login_page():
    st.title("💬 My Chatbot")
    login_tab, register_tab = st.tabs(["Log in", "Register"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
            if submitted:
                user_id = auth.login_user(username, password)
                if user_id:
                    _start_session(user_id, username.strip())
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            submitted = st.form_submit_button("Register")
            if submitted:
                success, message = auth.register_user(new_username, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)


# ---------- sidebar: conversation list + new chat + logout ----------

def render_sidebar():
    with st.sidebar:
        st.write(f"Logged in as **{st.session_state.username}**")

        if st.button("➕ New chat", use_container_width=True):
            new_id = db.create_conversation(st.session_state.user_id)
            st.session_state.conversation_id = new_id
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Your conversations")

        conversations = db.get_conversations(st.session_state.user_id)
        for conv in conversations:
            is_active = conv["id"] == st.session_state.get("conversation_id")
            label = ("👉 " if is_active else "") + conv["title"]
            if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                st.session_state.conversation_id = conv["id"]
                st.session_state.messages = db.get_messages(conv["id"])
                st.rerun()

        st.divider()
        if st.button("Log out", use_container_width=True):
            _end_session()
            st.rerun()


# ---------- main chat area ----------

def render_chat():
    st.title("💬 My Chatbot")

    if "conversation_id" not in st.session_state:
        conversations = db.get_conversations(st.session_state.user_id)
        if conversations:
            st.session_state.conversation_id = conversations[0]["id"]
            st.session_state.messages = db.get_messages(conversations[0]["id"])
        else:
            new_id = db.create_conversation(st.session_state.user_id)
            st.session_state.conversation_id = new_id
            st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        db.add_message(st.session_state.conversation_id, "user", user_input)
        db.rename_conversation_if_default(st.session_state.conversation_id, user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chat.get_answer(user_input, st.session_state.messages[:-1])
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        db.add_message(st.session_state.conversation_id, "assistant", answer)


# ---------- entrypoint ----------

if "user_id" not in st.session_state:
    render_login_page()
else:
    render_sidebar()
    render_chat()
