from langgraph.graph import END, START, StateGraph
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

langfuse = Langfuse(
  secret_key="sk-lf-3aefe75a-b11d-44e9-b6b0-8c33ab4b16cd",
  public_key="pk-lf-fd9986aa-7154-4de8-8497-ff9de62fa358",
  host="http://localhost:3000"
) # OR langfuse = get_client()
langfuse_handler = CallbackHandler()
# langfuse_handler = CallbackHandler(debug=True) #with trace ingestion.

from functions import (
    StateSchema,
    add,
    collect,
    divide,
    is_welcome_needed,
    multiply,
    subtract,
    welcome,
    )

builder = StateGraph(
    state_schema=StateSchema
    )
builder.add_node("add", add)
builder.add_node("divide", divide)
builder.add_node("multiply", multiply)
builder.add_node("subtract", subtract)
builder.add_node("collect", collect)
builder.add_node("welcome", welcome)
# add edge
builder.add_edge(START, "add")
builder.add_edge(START, "divide")
builder.add_edge(START, "multiply")
builder.add_edge(START, "subtract")
builder.add_edge("add", "collect")
builder.add_edge("divide", "collect")
builder.add_edge("multiply", "collect")
builder.add_edge("subtract", "collect")
builder.add_conditional_edges(
    "collect",
    is_welcome_needed,
    {"welcome": "welcome", END: END}
    )
builder.add_edge("collect", END)
builder.add_edge("welcome", END)
graph = builder.compile()

state = StateSchema(
    a=20,
    b=8,
    name="Alan",
    welcome_msg=str(),
    add_result=float(),
    div_result=float(),
    mul_result=float(),
    sub_result=float(),
    final_result=dict()
    )
result = graph.invoke(
    state,
    config={
        "callbacks": [langfuse_handler]
        # "run_id": uuid.uuid4()(),
        },
    verbose=True
    )
langfuse.flush()
print(result)
