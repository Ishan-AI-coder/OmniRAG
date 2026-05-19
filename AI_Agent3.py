from typing import Annotated,Sequence,TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage,BaseMessage,SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

# Annotated->provides additional info without affecting the type itself 
#eg:->email=Annotated(str,"This needs to be in proper email format ")
#print(email.__metadata__)

# Sequence->To automatically handle the state sequences such as by addding messages to the chat history
            # ->To avoid list manipulations 


#  BaseMessage->The foundational class for all the messages in Langgraph
#  ToolMessage->Passes data back to the LLM after its  call such as the content 
#  SystemMessage->Message for providing instructions to the LLM

#  add_messages->A kind of Reducer function
#Reducer function
# ->Rule that controls how updates from nodes are combined with the existing state
#->Tells us how to merge the current data with the existing data

load_dotenv()

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]


@tool
def add(a:int,b:int):
    """This is an addition function that adds two no.s together"""
    return a+b
@tool
def subtract(a:int,b:int):
    """This is a subtraction function that subtracts the two no.s """
    return a-b
@tool
def multiply(a:int,b:int):
    """This is a multiplication function that multiplies two no.s together"""
    return a*b

tools=[add,subtract,multiply]

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

def Model_call(state:AgentState)->AgentState:
    system_prompt=SystemMessage(content="You are my AI Assistant .Please answer my query to the best of your ability")
    response=model.invoke([system_prompt]+state["messages"])
    return {"messages":[response]}

def should_continue(state:AgentState):
    messages=state["messages"]
    lastmessage=messages[-1]
    if not lastmessage.tool_calls:
        return "end"
    else:
        return "continue"
    
graph=StateGraph(AgentState)
graph.add_node("AI_Agent",Model_call)

tool_node=ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.set_entry_point("AI_Agent")

graph.add_conditional_edges(
    "AI_Agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
)
graph.add_edge("tools","AI_Agent")

app=graph.compile()

from IPython.display import Image,display
display(Image(app.get_graph().draw_mermaid_png()))

# def print_stream(stream) :
#     for s in stream:
#         message=s["messages"][-1]
#         if isinstance(message,tuple):
#             print(message)
#         else:
#             message.pretty_print()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]

        if isinstance(message, HumanMessage):
            message.pretty_print()

        elif isinstance(message, AIMessage):
            if message.content:   # 🔑 KEY LINE
                message.pretty_print()

input={"messages":[("user","Add 3+4.Add 12+90.Subtract 12 and 34.Multiply 23 and 7")]}
print_stream(app.stream(input,stream_mode="values"))