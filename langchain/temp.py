from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="AI Assistant", page_icon=":shark:")
st.title("Welcome, I am AI Assistant :shark:")

# ---------------- Custom Streaming Callback ----------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.text_box = container.empty()

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.text_box.markdown(self.text + "▌")  # Cursor effect

    def on_llm_end(self, *args, **kwargs):
        self.text_box.markdown(self.text)  # Remove cursor

# ---------------- Model and Prompt ----------------
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", disable_streaming=False)

SYS_PROMPT = """Act as a helpful assistant and answer questions to the best of your ability. 
Do not make up answers."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYS_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | model

# ---------------- History ----------------
history = StreamlitChatMessageHistory()

# Show initial AI message if no history
if len(history.messages) == 0:
    history.add_ai_message("How can I help you?")

# Render full chat history
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

# ---------------- Chat Input ----------------
if user_input := st.chat_input():
    # Show user message
    st.chat_message("human").write(user_input)
    history.add_user_message(user_input)

    # Show streaming AI message
    with st.chat_message("ai"):
        handler = StreamHandler(st.container())

        # Invoke chain with history
        chain.invoke(
            {
                "input": user_input,
                "history": history.messages
            },
            config={"callbacks": [handler]}
        )

        # Add th
