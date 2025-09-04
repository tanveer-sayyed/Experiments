import streamlit as st
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain_core.callbacks import BaseCallbackHandler

# Create a custom callback handler
class StreamlitStreamingCallback(BaseCallbackHandler):
    def __init__(self):
        self.placeholder = None
        self.full_response = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token)
        self.full_response += token
        if self.placeholder: self.placeholder.markdown(self.full_response)
        else:
            self.placeholder = st.empty()
            self.placeholder.markdown(self.full_response)

st.title("Streaming Demo using Callbacks")
user_input = st.text_input("Ask me anything:")
streaming_callback = StreamlitStreamingCallback()

# Define a "stateless" schema
class AgentState(TypedDict):
    messages: List[str]
    streaming_content: str

# Define the LLM node with streaming
def call_llm(state: AgentState) -> Dict[str, Any]:
    """Node that calls the LLM with streaming"""
    llm = Ollama(
        model="mistral:7b",
        base_url="http://localhost:11435",
        callbacks=[streaming_callback] # enables live streaming
    )
    messages = state["messages"]
    last_message = messages[-1]["content"]
    response = llm.invoke(last_message)
    messages.append({"role": "assistant", "content": response})
    return {"messages": messages, "streaming_content": response}

# Create the graph
workflow = StateGraph(AgentState)
workflow.add_node("llm", call_llm)
workflow.set_entry_point("llm")
workflow.add_edge("llm", END)
app = workflow.compile()

if user_input:
    state = {
        "messages": [{"role": "user", "content": user_input}],
        "streaming_content": ""
    }
    app.invoke(state)


#############################################
# llm = ChatOpenAI()                        #
# chain = prompt | llm | StrOutputParser()  #
# return chain.stream({                     #
#     "chat_history": chat_history,         #
#     "user_question": user_query,          #
# })                                        #
#############################################