# Introduction
In the previous [post](https://harc007.github.io/swe_assistant/2024/11/12/BigCodeBench.html), we saw how BigCodeBench is a better dataset for evaluating code assistants. We also saw how prompt engineering results, though logically sound, have some simple syntax errors that need to be resolved. How do we move from prompt engineering to building such agents? Welcome to the world of Agentic AI.
# Agentic AI
Our goal is now to build a coding agent that can generate code, check for errors, test itself ad make edits to code if it fails. This process is very similar to how most programmers code. For this purpose, we will use Langgraph - a low-level orchestration framework for building controllable agents. For this post, let's begin by understanding and building the 'hello world' of the langgraph agent. This post will not evaluate the agent on bigcodebench as this post will build a agentic wrapper around prompt engineering. We will definitely build a more comprehensive agent in the next few posts and test it on bigcodebench.
# Basic Codebot
## Graph
We'll first create a basic codebot using langgraph. This codebot will respond directly to user messages. Start by creating a *StateGraph*. A StateGraph object defines the structure of our codebot as a *state machine*. We'll add *nodes* to represent the agent to specify how the bot should transition between these functions.
```
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
```
## Nodes
Next, add a *codebot* node. Nodes represent units of work. They are typically regular python functions.
```
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

def codebot(state: State):
    return {"messages": [llm.invoke(state['messages'])]}

graph_builder.add_node("codebot", codebot) 
```
Notice how the *codebot* node function takes the current *State* as input and returns a dictionary containing updated messages list under the key "messages". This is the basic pattern for all Langgraph node functions. The *add_messages* funtion in out State will append the LLM's response messages to whatever messages are already in the state.
## Edges
Next, add an *entry* point. This tells our graph where to start its work each time we run it. 
```
graph_builder.add_node(START, "codebot")
```
Similarly, set a *finish* point. This instructs the graph any time this node is run, you can exit.
```
graph_builder.add_node("codebot", END)
```
## Compile Graph
Finally, we'll want to be able to run our graph. To do so, call "compile()" on the graph builder. This creates a *CompiledGraph*, we can invoke on our state.
```
graph = graph_builder.compile()
```
You can visualize the graph using the *get_graph* method and one of the draw methods.
```
from Ipython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```
## Execution
```
def stream_graph_updates(user_input: str):
    for event in graph.stream({
        "messages": [{
                "role":"user", 
                "content":user_input
            }]
    }):
        for value in event.values():
            print("Assistant: ", value['messages'][-1].content)

while True:
    user_input = input("User: ")
    if user_input in ["quit", "q", "exit"]:
        print("Goodbye!")
        break
    stream_graph_updates(user_input)
```
When I passed input as 'create a function with inputs as name and place which are string variables. The function should return 'Hi <name>, I love <place>', the output was as follows
```
Assistant: 
def greet(name, place):
    return f"Hi {name}, I love your {place}"
```
# Conclusion
In this post, we really have put an agentic wrapper around an LLM. This post also introduces concept of *nodes* and *edges* integral to Langgraph which underpins the orchestration of the agent. We will add more complexity to this coding agent in future posts. Until then, catch a breath!