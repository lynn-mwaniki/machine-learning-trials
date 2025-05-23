from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

model =init_chat_model(model="gemini-2.0-flash",
                 api_key=gemini_key,
                 model_provider="google_genai")

class AssignmentClassifier(BaseModel):
    subject_type : Literal["math", "history", "language","science" ] = Field(
        ...,
        description="Classify the user message based on subject: Math, Science, History, or Language."
    )

class State(TypedDict):
    messages : Annotated[list, add_messages]
    subject_type :str | None


def SubjectClassifier(state:State):
    last_message = state["messages"][-1]
    classify_model = model.with_structured_output(AssignmentClassifier)
    result = classify_model.invoke([
        {
            "role": "system",
            "content":"""Classify the user message as:
            - 'math': If it involves calculations, numbers, equations, or problem-solving.
            - 'science': If it asks about physics, chemistry, biology, or technology concepts.
            - 'history': If it asks about historical events, figures, or timelines.
            - 'language': If it asks about grammar, translations, vocabulary, or writing help."""
        },{
            "role":"user", "content": last_message.content
        }
    ])
    return {"subject_type": result.subject_type}

def router(state:State):
    subject_type = state.get("subject_type", "math")  # Default to math if unsure
    return {"next": subject_type}

def math_agent(state:State):
    last_message=state["messages"][-1]
    messages=[
       {
           "role":"system",
           "content":"You are a Math Tutor. Solve equations, explain concepts, and provide step-by-step solutions."
       },
       {
           "role":"user",
           "content":last_message.content
       }
    ]
    reply = model.invoke(messages)
    return {"messages":[{"role": "assistant", "content": reply.content}]}


def science_agent(state:State):
    last_message=state["messages"][-1]
    messages=[
       {
           "role":"system",
           "content":"You are a Science Tutor. Answer science-related questions with clarity and examples."
       },
       {
           "role":"user",
           "content":last_message.content
       }
    ]
    reply = model.invoke(messages)
    return {"messages":[{"role": "assistant", "content": reply.content}]}

def language_agent(state:State):
    last_message=state["messages"][-1]
    messages=[
       {
           "role":"system",
           "content":"You are a Language Tutor. Help with grammar, vocabulary, and translations."
       },
       {
           "role":"user",
           "content":last_message.content
       }
    ]
    reply = model.invoke(messages)
    return {"messages":[{"role": "assistant", "content": reply.content}]}


def history_agent(state:State):
    last_message=state["messages"][-1]
    messages=[
       {
           "role":"system",
           "content":"You are a History Tutor. Provide detailed historical information and context for events."
       },
       {
           "role":"user",
           "content":last_message.content
       }
    ]
    reply = model.invoke(messages)
    return {"messages":[{"role": "assistant", "content": reply.content}]}



graph_builder = StateGraph(State)

graph_builder.add_node("classifier", SubjectClassifier)
graph_builder.add_node("router", router)
graph_builder.add_node("math", math_agent)
graph_builder.add_node("science", science_agent)
graph_builder.add_node("language", language_agent)
graph_builder.add_node("history", history_agent)


graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges("router",
                            lambda state: state.get("next"),
                            {"math": "math", "science": "science", "language": "language", "history": "history" }        
                                    )
graph_builder.add_edge("math", END)
graph_builder.add_edge("science", END)
graph_builder.add_edge("language", END)
graph_builder.add_edge("history", END)


graph = graph_builder.compile()


def run_tutor_bot():
    state = {"messages":[], "subject_type": None}

    while True:
        user_input = input("Ask message: ")
        if user_input.lower() == "exit":
            print("bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
            ]
        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"AI Tutor: {last_message.content}")

if __name__ == "__main__":
    run_tutor_bot()
