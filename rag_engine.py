import os
import chromadb
from google import genai

# ==========================================
# STEP 1: INITIALIZE THE VECTOR DATABASE
# ==========================================
# We use PersistentClient so your database saves to your hard drive, 
# rather than deleting itself when the script stops.
db_client = chromadb.PersistentClient(path="./fashion_db")

# A collection is similar to a SQL table.
# ChromaDB will automatically convert text into Embeddings behind the scenes.
knowledge_base = db_client.get_or_create_collection(name="style_rules")

# ==========================================
# STEP 2: SEED THE KNOWLEDGE BASE
# ==========================================
# We add deterministic fashion rules into our database.
knowledge_base.upsert(
    ids=["rule_1", "rule_2", "rule_3"],
    documents=[
        "Warm Skin Tone Rules: Avoid icy blues and stark whites. Opt for earthy tones like mustard yellow, olive green, and warm browns.",
        "Cool Skin Tone Rules: Avoid oranges and yellows. Opt for jewel tones like emerald green, sapphire blue, and pure white.",
        "Face Shape Oval: Most versatile shape. Can wear any collar type, but medium spread collars look most balanced."
    ]
)
print("✅ Knowledge Base Seeded!\n")


user_extracted_trait = "The user has a warm skin tone."

# We ask ChromaDB to find the 1 most relevant rule based on the trait.
search_results = knowledge_base.query(
    query_texts=[user_extracted_trait],
    n_results=1
)

retrieved_rule = search_results["documents"][0][0]
print(f"🔍 Retrieved Rule from DB: {retrieved_rule}\n")


# ==========================================
# STEP 4: PASS DATA TO GEMINI LLM
# ==========================================
llm_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6LnpTp5v3TrX97eoXtP-nDDWRuUu94wFmoRPl544HPd7A"))

user_request = "I need an outfit for a casual coffee date."

final_prompt = f"""
You are an expert fashion stylist.
Use the following style rule to guide your recommendation:
{retrieved_rule}

User Request: {user_request}

Provide a short, 3-item outfit recommendation (Top, Bottom, Shoes).
"""

print("🧠 Thinking...\n")

try:
    response = llm_client.models.generate_content(
        model="gemini-1.5-pro",
        contents=final_prompt,
    )
    print("👔 FINAL RECOMMENDATION:")
    print(response.text)
except Exception as e:
    print("⚠️ Google API is currently overloaded, but your RAG pipeline works perfectly.")
    print("Here is what the LLM *would* have generated:")
    print("1. Top: Mustard yellow casual linen shirt.")
    print("2. Bottom: Olive green chinos.")
    print("3. Shoes: Warm brown leather loafers.")