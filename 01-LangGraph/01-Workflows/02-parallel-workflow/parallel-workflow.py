from typing import TypedDict, Annotated
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


load_dotenv()

def merge_score_dicts(existing_dict: dict, new_dict: dict) -> dict:  #Reducer
    if existing_dict is None:
        return new_dict
    else:
        return{**existing_dict, **new_dict}

class State(TypedDict):
    raw_text : str
    safety_score : Annotated[dict[str,int],merge_score_dicts]
    


llm = ChatMistralAI(model = 'open-mistral-7b',temperature=0.4)

def toxicity_node(state: State) -> dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n {state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
    return{"safety_score":{"toxicity_level": score}}


def copyright_node(state: State) -> dict:
    print("\n [Branch 2] Analyzing Copyright & Originality Risks...")
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
            score = int(response.content.strip())
    except ValueError:
            score = 0
    return{"safety_score":{"copyright_risk": score}}


def culture_node(state: State) -> dict:
    print("\n [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        

    return {"safety_score": {"cultural_insensitivity": score}}

graph = StateGraph(State)

graph.add_node("toxicity",toxicity_node)
graph.add_node("copyright",copyright_node)
graph.add_node("culture",culture_node)


graph.add_edge(START,"toxicity")
graph.add_edge(START,"copyright")
graph.add_edge(START,"culture")

graph.add_edge("toxicity", END)
graph.add_edge("copyright", END)
graph.add_edge("culture", END)

app = graph.compile()

sample_script = """
    Yo guys! Welcome back to the stream. Today I am going to show you how to hack into 
    your friend's system using a script I copied directly from an online forum. 
    Honestly, traditional security protocols are absolute garbage and anyone still using 
    them is an absolute idiot. Let's dive into the code!
    """
    
result = app.invoke({
    "raw_text": sample_script,
    "safety_score":{}
})

print(result["safety_score"])

