# Introduction
In the previous [post](https://harc007.github.io/swe_assistant/2025/03/19/AgenticFlowIntro.html), we saw building blocks of a langgraph agent which are nodes and edges. Today, we are going to build on that further and create a slightly more complex agent which can filter out tasks that are not a coding task. If it is a coding task, it goes on to generate the code. Again, this is just an extension to the previous post. The idea of making this a separate post is to understand how to add multiple sub-agents to the overall agent.

# Architecture
![alt text](/swe_assistant/docs/assets/lg_01.JPG "agent_representation")
As you can see we want to first check if the input task is a coding task. If not, we do not want this agent to answer this question. And if it is, we want the LLM to try and answer it.
```
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    is_code_task: bool

graph_builder = StateGraph(State)
```
We have now added a new parameter to the State of the graph to track if this is a code task or not. That is a functionality required and it is always a good practice to keep a track of a functionality status in the state.

```
from langchain_openai.chat_models import ChatOpenAI
from typing import Literal

llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

def codebot(state: State):
    return {"messages": [llm.invoke(state["messages"])], "is_code_task":True}

def check_if_code_task(state: State):
    code_task_prompt = f"""Check if the user is asking you to write python code for a specific problem within ``? `{state["messages"]}`
                            Return your answer as END if it is not a coding task. Return CODEBOT if it is a coding task. 
                            Remember that answer should only be 1 word."""
    response = llm.invoke(code_task_prompt)
    if response.content == "END":
        print("Assistant: This is not a python code generation request")
        return {"messages": [response], "is_code_task":False}
    return {"messages": [response], "is_code_task":True}

def should_continue(state:State)->Literal["codebot", END]:
    if state["is_code_task"]:
        return "codebot"
    return END

graph_builder.add_node("check_if_code_task", check_if_code_task)
graph_builder.add_node("codebot", codebot)
```

Compared to the previous post, we have now added two new functions. One will check if task is a coding task or not. This requires us to create a custom prompt for the task. The other one will decipher if we should pass the input to the sub-agent that creates code or just end. This function does not need the support of the LLM, but it only has to be aware of the state.

# Nodes and Edges
This warrants the question - how to decipher what is a node and what is a edge? A simple way to think about this is to see the functions where you want to change the state. Those become your nodes. And the functions that take a call based on the state become your edges. Let's build out the graph:
```
graph_builder.add_edge(START, "check_if_code_task")
graph_builder.add_conditional_edges("check_if_code_task", should_continue)
graph_builder.add_edge("codebot", END)
```

So the graph first checks if it is a coding task and then based on whether it is or not goes tho the code generating bot or ends.

```
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({
        "messages": [{
            "role": "user", 
            "content": user_input
        }], 
        "is_code_task": False
    }):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        break
```

# Conclusion
You can try passing various coding requests and understand if the abovve code works at code generation when it is supposed to and does nothing when it does not. We have taken another step closer to a full blown coding agent. Let's see if we can create our own in the next blog. Until then, relax and stretch out!