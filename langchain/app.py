from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from operator import itemgetter
import streamlit as st

# initial app landing page
st.set_page_config(page_title="AI Assistant", page_icon=":shark:")
st.title("Welcome i am AI Assistant :shark:")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)

model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", disable_streaming = False)

SYS_PROMPT = """Act as helpful assistant and answer questions to the best of your ability 
                Do not make up answers.
             """

prompt = ChatPromptTemplate.from_messages([
    ("system", SYS_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm_chain = prompt | model
streamlit_message_history = StreamlitChatMessageHistory()

conversation_chain = RunnableWithMessageHistory(
    llm_chain, 
    lambda session_id: streamlit_message_history,
    input_messages_key="input",
    history_messages_key="history"
)

if len(streamlit_message_history.messages) == 0:
    streamlit_message_history.add_ai_message("How can i help you?")

for msg in streamlit_message_history.messages:
    st.chat_message(msg.type).write(msg.content)

if user_prompt := st.chat_input():
    st.chat_message("human").write(user_prompt)

    with st.chat_message("ai"):

        streamlit_handler = StreamHandler(st.empty())
        config = {"configurable": {"session_id": "any"},
                  "callbacks": [streamlit_handler]}
        
        response = conversation_chain.invoke({"input": user_prompt}, config)
        streamlit_message_history.add_ai_message(response.content)

