from typing import TypedDict,List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class AgentState(TypedDict):
    messages:List[HumanMessage]


def process(state:AgentState)->AgentState:
    """This ai model caters to the users requests"""
    response=llm.invoke(state["messages"])
    print(response.content)



graph=StateGraph(AgentState)
graph.add_node("Agent",process)
graph.add_edge(START,"Agent")
graph.add_edge("Agent",END)
app=graph.compile()


user_input=input("Enter :")
while user_input!="exit":
    app.invoke({"messages":[HumanMessage(content=user_input)]})
    user_input=input("Enter:")


