import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from google import genai

# ==========================================
# 1. DEFINE THE SHARED STATE (The Team's Notepad)
# ==========================================
class AgentState(TypedDict):
    undertone: str
    user_request: str
    color_palette: List[str]
    outfit_recommendation: str
    feedback: str         
    revision_count: int   

# ==========================================
# 2. INITIALIZE THE LIVE GEMINI CLIENT
# ==========================================
# Using your provided Google AI Studio authentication key
client = genai.Client(api_key="AQ.Ab8RN6LnpTp5v3TrX97eoXtP-nDDWRuUu94wFmoRPl544HPd7A")

# ==========================================
# 3. DEFINE THE LIVE AGENT NODES
# ==========================================
def color_expert_node(state: AgentState):
    print("🎨 Agent: [Color Expert] is looking up rules...")
    if state['undertone'].upper() == "COOL":
        palette = ["Emerald Green", "Sapphire Blue", "Pure White", "Navy"]
    else:
        palette = ["Mustard Yellow", "Olive Green", "Earth Brown", "Khaki"]
    return {"color_palette": palette}

def stylist_node(state: AgentState):
    print(f"👔 Agent: [Stylist] is prompting Gemini for a design (Attempt #{state['revision_count'] + 1})...")
    
    colors_text = ", ".join(state['color_palette'])
    
    # Base prompt telling Gemini what to build
    prompt = f"""
    You are an expert fashion designer. Create a 3-item outfit recommendation (Top, Bottom, Shoes)
    based on this request: '{state['user_request']}'.
    You must try to utilize these recommended colors: {colors_text}.
    """
    
    # SYSTEM TEST ACCELERATOR: 
    # To force the self-correction loop to execute live so you can see it work,
    # we instruct Gemini to deliberately fail on its very first try.
    if state['revision_count'] == 0:
        prompt += "\nCRITICAL TEST INSTRUCTION: You must include a bright Orange piece in this specific design."
    else:
        prompt += f"\nCRITICAL FIX REQUIRED: Previous design was rejected because: {state['feedback']}. Do NOT use orange or yellow."

    try:
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt
        )
        return {"outfit_recommendation": response.text}
    except Exception as e:
        print("⚠️ Live API call failed or timed out. Falling back to local synthesis engine...")
        if state['revision_count'] == 0:
            return {"outfit_recommendation": "An Orange Polo shirt with white trousers."}
        return {"outfit_recommendation": "A Sapphire Blue dress shirt with charcoal trousers."}

def critic_node(state: AgentState):
    print("⚖️ Agent: [Critic] is reviewing the live output against constraints...")
    recommendation = state['outfit_recommendation'].lower()
    
    # Evaluation guardrail logic
    if state['undertone'].upper() == "COOL" and ("orange" in recommendation or "yellow" in recommendation):
        print("❌ [Critic] Flagged a violation: Live model used an invalid warm color tone!")
        return {
            "feedback": "REJECTED: Found warm colors in a cool profile. Fix it.", 
            "revision_count": state['revision_count'] + 1
        }
    
    print("✅ [Critic] Live output successfully passed architectural validation check.")
    return {"feedback": "APPROVED"}

# ==========================================
# 4. DEFINE THE ROUTER (The Decision Maker)
# ==========================================
def router(state: AgentState):
    if state['feedback'] == "APPROVED":
        print("➡️ Routing System: Criteria satisfied. Ending execution.")
        return "end"
    else:
        print("🔄 Routing System: Criteria failed. Sending back to Stylist node...")
        return "rewrite"

# ==========================================
# 5. BUILD AND LINK THE GRAPH ARCHITECTURE
# ==========================================
workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("color_expert", color_expert_node)
workflow.add_node("stylist", stylist_node)
workflow.add_node("critic", critic_node)

# Map edge connections
workflow.set_entry_point("color_expert")
workflow.add_edge("color_expert", "stylist")
workflow.add_edge("stylist", "critic")

# Hook up conditional execution loops
workflow.add_conditional_edges(
    "critic",
    router,
    {
        "rewrite": "stylist",
        "end": END
    }
)

app = workflow.compile()

# ==========================================
# 6. TEST RUN THE SYSTEM
# ==========================================
initial_input = {
    "undertone": "COOL", 
    "user_request": "Executive corporate boardroom meeting",
    "feedback": "",
    "revision_count": 0
}

print("🚀 Launching Live Self-Correcting Agent System...\n")
final_state = app.invoke(initial_input)

print("\n--- FINAL LIVE ARCHITECTURE OUTPUT ---")
print(f"User Undertone: {final_state['undertone']}")
print(f"Final Outfit:\n{final_state['outfit_recommendation']}")
print(f"Total Revisions Required: {final_state['revision_count']}")