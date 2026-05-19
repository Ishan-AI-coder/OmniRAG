from typing import Annotated,Sequence,TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage,BaseMessage,SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

# This is the document variable that stores the document content
document_content=""


class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]

@tool
def update(content:str)->str:
    """Updates the document with the provided content"""
    global document_content 
    document_content=content

    return f"Document has been updated succesfully! The current content is \n {document_content}"

@tool
def save(filename:str)->str:
    """Save the current document to a text file and finish the process
    
    Args:
        filename :Name for the text file.
    """
    global document_content
    if not filename.endswith('.txt'):
        filename=f"{filename}.txt"

    try:
        with open(filename,"w") as file:
            file.write(document_content)
            print(f"\nDocument has been successfully saved to :{filename}")
            return f"Document has been saved to '{filename}'. "
        
    except Exception as e:
        return f"\n Error in saving document:{str(e)}"
    
tools=[update,save]

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools()

def Our_agent(state:AgentState)->AgentState:
    system_prompt = SystemMessage(
    content=f"""
                You are **Drafter**, a precise and helpful writing assistant.

Your responsibility is to help the user **edit, update, and finalize a document**.

### Core Rules
1. You must always work with the **current document content** provided below.
2. You must **never invent or assume missing content**.
3. You must clearly understand the user’s intent before acting.

### Tool Usage Rules
- If the user asks to **edit, update, rewrite, add, remove, or modify** any part of the document:
  → You MUST call the **`update` tool**.
  → The tool input MUST contain the **entire updated document**, not just the changed section.

- If the user says they are **done**, wants to **save**, **finish**, or **finalize**:
  → You MUST call the **`save` tool**.
  → Do NOT make further edits unless explicitly requested.

- If the user asks a **question or clarification** without requesting a change:
  → Respond normally WITHOUT calling any tool.

### Output Requirements
- After every successful update, you MUST show the **full current document state**.
- Do NOT describe what you changed unless the user asks.
- Do NOT include tool instructions or internal reasoning in the final response.

### Current Document Content
{document_content}
"""
)
    if not state["messages"]:
        user_input="I'm ready to help you update a document .What would you like to create?"
        user_message=HumanMessage(content=user_input)
    else:
        user_input= input("\n What would you like to do with the document?")
        print(f"\n👤 USER:{user_input}")
        user_message=HumanMessage(content=user_input)
    
    all_messages=[system_prompt]+list(state["messages"])+[user_message]

    response=model.invoke(all_messages)
    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🛠 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]

    if not messages:
        return "continue"

    # This looks for the most recent tool message.....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (
            isinstance(message, ToolMessage)
            and "saved" in message.content.lower()
            and "document" in message.content.lower()
        ):
            return "end"  # goes to the end edge which leads to the endpoint
    return "continue"


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠 TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", Our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()

def run_document_agent():
    print("\n===== DRAFTER =====")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
