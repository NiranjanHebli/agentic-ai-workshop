import os, json 
from dotenv import load_dotenv 
from typing import TypedDict, List 
from langgraph.graph import StateGraph, START, END 
from langchain_core.messages import SystemMessage, HumanMessage 
from langchain_community.tools import DuckDuckGoSearchRun 
from langchain_groq import ChatGroq 

load_dotenv()
# Shared LLM Instance 
API_KEY = os.environ["GROQ_API_KEY"]
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=API_KEY) 

class AgentState(TypedDict):
    goal: str 
    tasks: List[str] 
    results: List[str] 
    critique: str 
    approved: bool 
    iterations: int 
    score: float 

# 2. Define the Tools 
search = DuckDuckGoSearchRun()

def planner(state: AgentState) -> AgentState: 
    system = """You are a planning agent. Break the user's goal into at most 5 concrete, actionable tasks. Respond ONLY with a valid JSON array of strings. No preamble, No Markdown."""

    messages = [SystemMessage(content=system), HumanMessage(content=state["goal"])] 
    response = llm.invoke(messages).content.strip()
    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        tasks = json.loads(clean) 
    except json.JSONDecodeError:
        tasks = [response]

    print(f"\n[Planner] Generated {len(tasks)} tasks")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")

    return {**state, "tasks": tasks}

def executor(state: AgentState) -> AgentState:
    results = []
    critique_ctx = ""
    if state["critique"]:
        critique_ctx = f"\n\n Your Previous attempt was rejected. Critique: {state['critique']}\n Improve your output accordingly "

    for task in state["tasks"]:
        system = f"""You are an execution agent. Complete the task below thoroughly 
        .Use web search if you need current information. {critique_ctx}"""

        # try web search for research tasks 
        search_ctx = ""
        try:
            search_result = search.run(task[:100])
            search_ctx = f"\n\n Web search result for context: \n {search_result[:800]}" 
        except Exception as e:
            print(f"\n [Execution Agent] Search failed for task {task}: {e}")

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Task: {task} {search_ctx} ")
        ]
        response = llm.invoke(messages)
        results.append(response.content.strip())
        print(f"\n [Execution Agent] Completed Task: {task}")
    
    return {**state, "results": results, "iterations": state["iterations"]+1} 

def verifier(state: AgentState) -> AgentState:
    if state["iterations"] >= 3:
        print("\n[Verifier] Max iterations reached. force approving.")
        return {**state, "approved": True}

    combined_results = '\n\n'.join(
        f"Task {i+1}: {task}\n Result: {result}\n" 
        for i, (task, result) in enumerate(zip(state['tasks'], state['results']))
    )

    system = """You are a quality verifier. Evaluate the results against the original goal using this rubric:
    - Completeness: Does it fully address the goal? (0-0.4)
    - Accuracy: Is the information correct and specific? (0-0.3)
    - Clarity: Is it well-structured and clear? (0-0.3)
    Sum the scores for a total between 0.0 and 1.0 
    Response ONLY as JSON : {"score":0.85,"approved":true, "critique":"..."}"""

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Orginal goal : {state['goal']}\n\n Results: {combined_results}")

    ]
    raw = llm.invoke(messages).content.strip()
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        verdict = json.loads(clean)
        approved = verdict.get("approved", False)
        critique = verdict.get("critique","")
        score = verdict.get("score",0)
        
        # Enforce numeric threshold
        if score < 0.7:
            approved = False
            critique += f"\n[System Note: Score {score} is below the required 0.7 threshold.]"
            
    except:
        approved, critique, score = False, raw, 0.0
    print(f"\n[Verifier] Score: {score} | Approved: {approved}")
    return {**state, "approved": approved, "critique": critique, "score": score}

def should_continue(state: AgentState):
    if state["approved"]:
        return END
    return "executor"

# Graph Construction
graph = StateGraph(AgentState)

graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("verifier", verifier)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "verifier")
graph.add_conditional_edges("verifier", should_continue)
app = graph.compile()

initial_state : AgentState = {
    "goal": "Research and summarise the top 3 trends in generative AI for 2026",
    "tasks": [],
    "results": [],
    "critique": "",
    "approved": False,
    "iterations": 0,
    "score": 0.0
}

final_state = app.invoke(initial_state)

print("\n =================== Final Result ==================\n")
for i , (task, result) in enumerate(zip(final_state["tasks"], final_state["results"]),1):
    print(f"Task {i}: {task}")
    print(f"Result: {result}")
    print("-" * 60)

print(f"Completed in {final_state['iterations']} iterations")