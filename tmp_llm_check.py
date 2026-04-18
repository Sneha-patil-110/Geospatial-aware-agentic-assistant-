from app.assistant import run_assistant

print("start")
r = run_assistant(
    18.5204,
    73.8567,
    "Evacuation safety and relief camps near me?",
    district="Pune",
    state="Maharashtra",
    radius_m=5000,
    top_k_docs=5,
)
a = r.get("risk_analysis", {})
print("used_llm=", a.get("used_llm"))
print("risk_level=", a.get("risk_level"))
print("explanation=", (a.get("explanation") or "")[:180])
