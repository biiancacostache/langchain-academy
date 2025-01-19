# Standard library imports
from typing import TypedDict, NotRequired

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
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages


class FrenchJournalState(TypedDict):
    """State for the French Journal Assistant.
    
    Attributes:
        messages: List of conversation messages
        check_continue: Flag to control conversation flow
    """
    messages: list[BaseMessage]
    check_continue: NotRequired[bool]


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
    
    return chain.invoke({"text": text})

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

def should_continue(state: FrenchJournalState) -> FrenchJournalState:
    """Ask user if they want to continue journaling."""
     # Get the last message from the user if it exists
    messages = state["messages"]

    if messages and isinstance(messages[-1], HumanMessage):
        last_message = messages[-1].content.lower().strip()
        wants_to_continue = not (last_message in ["non", "no", "stop", "quit", "exit"])
    else:
        question = AIMessage(content="Voulez-vous continuer? (oui/non): ")
        messages.append(question)
        wants_to_continue = True

    return {
        "messages": messages,  # Preserve existing messages
        "check_continue": wants_to_continue
    }

def assistant(state: FrenchJournalState) -> FrenchJournalState:
    """Process messages and decide on next action."""
    print("Current state:", state)  # Debug print
    # Create a clean list of messages for the LLM
    messages = []
    
    # Add system message
    messages.append(SystemMessage(content=SYSTEM_PROMPT))
    
    # Add conversation history
    for msg in state["messages"]:
        if isinstance(msg, dict):
            # Convert dict to appropriate message type
            if msg["type"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                messages.append(AIMessage(content=msg["content"]))
        else:
            # Message is already a Message object
            messages.append(msg)
    
    # Get response from LLM
    response = llm_with_tools.invoke(messages)
    print("LLM Response:", response)  # Debug print
    
    return {
        "messages": state["messages"] + [response],
        "check_continue": state.get("check_continue", True)
    }

def continue_condition(state: FrenchJournalState) -> bool:
    """Check if the user wants to continue journaling."""
    return state["check_continue", True]

# Build graph
builder = StateGraph(FrenchJournalState)

# Add nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("should_continue", should_continue)

# Add edges
builder.add_edge(START, "assistant")
builder.add_edge("assistant", "tools")
builder.add_edge("tools", "assistant")
builder.add_edge("assistant", "should_continue")
builder.add_conditional_edges(
    "should_continue",
    continue_condition,
    {
        True: "assistant",  # Continue conversation
        False: END         # End the conversation
    }
)

# Compile graph
graph = builder.compile()