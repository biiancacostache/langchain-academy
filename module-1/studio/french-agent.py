# Standard library imports
from typing import TypedDict, NotRequired, Literal
import json

# Langchain imports
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# LangGraph specific imports
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

class FrenchJournalState(TypedDict):
    """State for the French Journal Assistant.
    
    Attributes:
        messages: List of conversation messages
        check_continue: Flag to control conversation flow
    """
    messages: list[BaseMessage]
    completed: NotRequired[bool]


@tool
def translate_from_english_to_french(text: str) -> str:
    """Translates a text from English to French.

    Args:
        text: the text to translate
    """
    
    translator = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Translate the following text from English to French: {text}\n\n"
        "Provide the translation in French only, and explain your reasoning in English."
    )

    chain = prompt | translator | StrOutputParser()
    result = chain.invoke({"text": text})
    
    return f"Translation: {result}"

@tool
def correct_grammar(text: str) -> str:
    """Corrects French grammar and vocabulary in the given text.

    Args:
        text: French text to correct
    """
    corrector = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "As a French language expert, review and correct the following French text. "
        "Provide the corrected version followed by a brief explanation of the changes in English:\n\n{text}"
    )
    
    chain = prompt | corrector | StrOutputParser()
    
    return chain.invoke({"text": text})

@tool
def generate_followup_questions(entry: str) -> str:
    """Generates thoughtful follow-up questions about the journal entry in French.

    Args:
        entry: The journal entry in French
    """
    questioner = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template(
        "Based on this French journal entry, generate 2-3 thoughtful follow-up questions in French "
        "that would encourage deeper reflection:\n\n{entry}"
    )
    
    chain = prompt | questioner | StrOutputParser()
    
    return chain.invoke({"entry": entry})

@tool
def suggest_enhancements(entry: str) -> str:
    """Suggests ways to enhance the journal entry with more detailed French vocabulary and expressions.

    Args:
        entry: The journal entry in French
    """
    enhancer = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template(
        "Review this French journal entry and suggest enhancements using more sophisticated French vocabulary "
        "and expressions. Include example phrases that could be incorporated:\n\n{entry}"
    )
    
    chain = prompt | enhancer | StrOutputParser()
    
    return chain.invoke({"entry": entry})

tools = [
    translate_from_english_to_french,
    correct_grammar,
    generate_followup_questions,
    suggest_enhancements
]

llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# System message
SYSTEM_PROMPT = """You are a French Journal Assistant, an interactive agent designed to help users log their daily journal entries in French. \
A human will talk to you about their day, and you will provide corrections to their French grammar and vocabulary, \
ask follow-up questions to encourage deeper reflection, and suggest additional notes or ideas to enhance their entries.

If the user submits their journal entry in English, first translate it into French calling translate, and present it back to them. \
Then, call correct_input to provide feedback on grammar and vocabulary. \
After correcting the entry, ask follow-up questions using ask_questions to prompt further reflection on their experiences. \
You can also suggest enhancements with suggest_enhancements based on the content of their entry.

Always ensure that your responses are encouraging and supportive, fostering a positive writing experience. \
If you need clarification on any entry or response from the user, ask a clarifying question before proceeding. \
Your goal is to help the user express themselves clearly and creatively in French.

Try to always respond in French. Use English only when the user requires more guidance for French."""

WELCOME_MSG = "Bienvenue dans l'Assistant de Journal en Français! Tapez `q` pour quitter. Quelle note de journal aimeriez-vous écrire aujourd'hui? Vous pouvez écrire en anglais ou en français."

def chatbot_with_tools_node(state: FrenchJournalState) -> FrenchJournalState:
    """Chatbot that can call tools."""
    defaults = {"messages": [], "completed": False}

    if not state["messages"]:
        return defaults | state | {"messages": [AIMessage(content=WELCOME_MSG)]}

    # Get the last message
    last_msg = state["messages"][-1]

    # Process only the last message if needed
    if isinstance(last_msg, tuple) and last_msg[0] == "user":
        state["messages"][-1] = HumanMessage(content=last_msg[1])

    # Create message list with system prompt and history
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    # Set up some defaults if not already set, then pass through the provided state,
    # overriding only the "messages" field.
    return defaults | state | {"messages": state["messages"] + [response]}

def human_node(state: FrenchJournalState) -> FrenchJournalState:
    """Writes in French or in English."""
    last_msg = state["messages"][-1]

    user_input = interrupt(value="Ready for user input.")

    if user_input in {"q", "quit", "exit", "goodbye", "au revoir"}:
        state["check_continue"] = False
    
    return  state | {"messages": [("user", user_input)]}

# def maybe_route_to_tools(state: FrenchJournalState) -> Literal["tools", "human"]:
#     """Route between human or tool nodes, depending if a tool call is made."""
#     if not (msgs := state.get("messages", [])):
#         raise ValueError(f"No messages in state: {state}")

#     # Only route based on the last message
#     last_msg = state["messages"][-1]

#     # When the chatbot returns tool_calls, route to the "tools" node.
#     if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
#         return "tools"
#     else:
#         return "human"

def maybe_exit_human(state: FrenchJournalState) -> Literal["chatbot", END]:
    """Check if the user wants to continue journaling."""
    if state["completed"]:
        return END
    else:
        return "chatbot"


# Build graph
builder = StateGraph(FrenchJournalState)

# Add nodes
builder.add_node("chatbot", chatbot_with_tools_node)
# builder.add_node("human", human_node)
builder.add_node("tools", ToolNode(tools, messages_key="messages"))

# Add edges
builder.add_edge(START, "chatbot")

builder.add_conditional_edges("chatbot", tools_condition)

# Human may go back to chatbot or exit
# builder.add_conditional_edges("human", maybe_exit_human)

# Tools always route back to the chat
builder.add_edge("tools", "chatbot")



# Compile graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)