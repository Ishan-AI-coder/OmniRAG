from typing import TypedDict,List,Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages:List[Union[AIMessage,HumanMessage]]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def process(state:AgentState)->AgentState:
    """This is a sample ai tool"""
    response=llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response))
    print(f"\nAI:{response.content}")
    return state

graph=StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent=graph.compile()


conversation_history=[]

user_input=input("Enter :")
while(user_input!="exit"):
    conversation_history.append((HumanMessage(content=user_input)))
    result=agent.invoke({"messages":[HumanMessage(content=user_input)]})
    conversation_history=result["messages"]
    user_input=input("Enter :")


with open("logging.txt","w") as file:
    file.write("Your conversation log\n")

    for message in conversation_history:
        if isinstance(message,HumanMessage):
            file.write(f"You:{message.content}")
        if isinstance(message,AIMessage):
            file.write(f"AI :{message.content}")

    file.write("The conversation history is stored\n")       
