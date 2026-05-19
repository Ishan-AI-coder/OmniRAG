import os
from typing import TypedDict,List,Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages:List[Union[HumanMessage,AIMessage]]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def process(state:AgentState)->AgentState:
    """This tool solves the user's queries"""
    response=llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
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
    conversation_history.append(HumanMessage(content=user_input))
    result =agent.invoke({"messages":conversation_history})
    conversation_history=result["messages"]
    user_input=input("Enter :")

with open("logging.txt","w") as file:
    file.write("Your conversational log :\n")
    
    for message in conversation_history:
        if isinstance(message,HumanMessage):
            file.write(f"You:{message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"AI:{message.content}\n")
    file.write("End of conversation\n")

print("Conversation saved to logging.txt")




#If you want to create a function that stores conversation over various chats .Just do this 
# from langchain_core.messages import HumanMessage, AIMessage
# import os

# def load_conversation():
#     messages = []
#     if os.path.exists("logging.txt"):
#         with open("logging.txt", "r") as file:
#             for line in file:
#                 if line.startswith("You:"):
#                     messages.append(
#                         HumanMessage(content=line.replace("You:", "").strip())
#                     )
#                 elif line.startswith("AI:"):
#                     messages.append(
#                         AIMessage(content=line.replace("AI:", "").strip())
#                     )
#     return messages


# And then replace 
# conversation_history=[]      with
#  
#conversation_history = load_conversation()
