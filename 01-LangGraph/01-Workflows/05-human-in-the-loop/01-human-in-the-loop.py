import os 
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()



writer_llm = ChatMistralAI(model = 'open-mistral-7b',temperature=0.7)



# State
class State(TypedDict):
    topic : str
    messages : Annotated[list,add_messages]
    draft : str
    review_feedback : str
    is_approved : bool
    attempts : int
  
WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Write engaging, professional "
    "LinkedIn posts about the given topic. "
    "Rules: strong hook in the first line, one clear takeaway, easy to skim "
    "with short paragraphs, roughly 150-200 words, end with an engaging "
    "question or CTA, no hashtags. "
    "If you receive feedback on a previous draft, address every point carefully."
)  

def writer_node(state: State) -> dict:
    """Writes (or rewrites) the LinkedIn post."""
    topic = state["topic"]
    attempt = state.get("attempts",0) + 1 
    feedback = state["review_feedback"]
    
    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic {topic}"
        )
    else:
        user_message = (
            f"your previous draft on '{topic}' was rejected"
            f"Here is the reviewer's feedback \n\n {feedback}\n\n"
            f"Write a new, improved draft that fixes every issue mentiond"
            f"do not repeat the same mistake"
        )
    messages = [
    ("system", WRITER_SYSTEM_PROMPT),
    ("human", user_message)
]
    response = writer_llm.invoke(messages)
    
    return{
        "messages" : [("human",user_message),response],
        "attempts" : attempt,
        "draft" : response.content  
    }
        
    



# def extract_draft_node(state: State) -> dict:
#     """After the writer finishes tool calls, pulls the final text out as the draft."""
    
#     draft = state["messages"][-1].content
#     print(f"\n\n generated post \n {draft} \n ")

#     return{
#         "draft" : draft
#     }
    
    

def human_review_node(state: State) -> dict:
    human_response = interrupt({
        "draft" : state["draft"],
        "attempts": state["attempts"],
        "instructions": "Type 'approved' to accept, or type your feedback to request a rewrite."
    })
    response = human_response.strip()
    
    if response.lower().strip() in ["approved", "approve", "yes", "ok", "good"]:
        return{
            "is_approved" : True,
            "review_feedback": "Approved by human"
        }
    
    else:
        return{
                    "is_approved" : False,
                    "review_feedback": response.strip()
        }
    
# Router Functions

# def should_use_tool(state: State):
#     last_message = state["messages"][-1]
    
#     if getattr(last_message, "tool_calls", None):
#         return "tool"
#     else:
#         return "extractor"
    
def should_stop_looping(state: State):
    if state["attempts"] >= 3:
        return END
    elif state["is_approved"]:
        return END
    else:
        return "writer"

# Build Graph

graph = StateGraph(State)

graph.add_node('writer',writer_node)
graph.add_node('review',human_review_node)


graph.add_edge(START,"writer")
graph.add_edge("writer","review")
graph.add_conditional_edges("review",should_stop_looping)
        
checkpoint = MemorySaver()
app = graph.compile(checkpointer=checkpoint)

print("=" * 55)
print("Welcome to the LinkedIn Post Generator")
print("=" * 55)
print("\nThis tool will draft a LinkedIn post for you, review it")
print("itself, and iterate until it's publish-ready.")

print("=" * 55)

topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

if not topic:
    print("\nNo topic given. Exiting.")
else:
    print("\nStarting generation...\n")

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempts": 0,
    }
    config = {"configurable": {"thread_id": "linkedin_session_1"}}

    final_state = app.invoke(initial_state, config=config)

    
    while "__interrupt__" in final_state:
        interrupt_data = final_state["__interrupt__"][0].value

        print("\n" + "=" * 55)
        print(f"DRAFT FOR YOUR REVIEW (Attempt {interrupt_data['attempts']})")
        print("=" * 55)
        print("Draft:\n", interrupt_data["draft"])
        print("=" * 55) 
        print(f"\n{interrupt_data['instructions']}")

        user_input = input("Enter your feedback:\n").strip()

        final_state = app.invoke(
            Command(resume=user_input),
            config=config
    )
        
        
    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(final_state["draft"])
    print("=" * 55)
    print(f"Total attempts: {final_state['attempts']}s")
    print(f"Approved: {final_state['is_approved']}")