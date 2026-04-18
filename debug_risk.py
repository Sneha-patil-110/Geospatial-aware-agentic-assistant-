#!/usr/bin/env python
"""Debug full risk response generation."""
import sys
import logging
from dotenv import load_dotenv

# Set up debug logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

load_dotenv()
sys.path.insert(0, '.')

from app.reasoning import generate_risk_response

# Simulate context from the assistant
test_context = {
    "structured_data": {
        "weather": {"temp": 28, "condition": "clear"},
        "hazard": {"flood_risk": "moderate"},
        "alerts": [],
        "history": ["flood_2023"]
    },
    "infra": {
        "category_counts": {"medical": 5, "shelter": 10, "water": 8},
        "safety_score": 0.65,
    },
    "risk_scoring": {
        "district": "Pune",
        "state": "Maharashtra", 
        "score": 0.58,
        "label": "Moderate"
    }
}

print('Testing Risk Response Generation...')
print('=' * 70)

try:
    result = generate_risk_response("Analyze hazard and infra risk for Pune", test_context)
    
    print('\n' + '=' * 70)
    print('RESULT:')
    print('=' * 70)
    print(f'Risk Level: {result.get("risk_level")}')
    print(f'Used LLM: {result.get("used_llm")}')
    print(f'Explanation: {result.get("explanation")[:100]}...')
    print(f'Action: {result.get("recommended_action")[:100]}...')
    
    if result.get("used_llm"):
        print('\n✓✓✓ SUCCESS - LLM IS BEING USED!')
    else:
        print('\n✗✗✗ FALLBACK USED - LLM DID NOT WORK')
        
except Exception as e:
    import traceback
    print(f'\n✗ ERROR: {e}')
    traceback.print_exc()
