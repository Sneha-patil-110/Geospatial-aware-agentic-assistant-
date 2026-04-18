#!/usr/bin/env python
"""Debug the LLM response for risk analysis."""
import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from app.config import cfg, nvidia_reasoning_api_key

# Set up logging
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

RISK_SYSTEM_PROMPT = (
    "You are a civilian safety and disaster risk assistant. "
    "Use ONLY the structured context provided in the user message. "
    "Do not invent facts. If data is missing, say so. "
    "Return ONLY valid JSON with keys: risk_level (Low|Medium|High), "
    "explanation (string), recommended_action (string)."
)

# Simulate a risk analysis prompt
test_prompt = """
Based on this hazard context:
- Weather: No active severe weather
- Hazards: Flood risk moderate
- Alerts: None
- History: Previous flood events in area

Infrastructure items nearby: 65
District risk score: 0.58 (Moderate)

Provide a risk assessment in JSON format.
"""

print('Testing LLM Risk Analysis...')
print('=' * 70)

try:
    api_key = nvidia_reasoning_api_key()
    print(f'API Key present: {bool(api_key)}')
    
    client = OpenAI(
        api_key=api_key,
        base_url='https://integrate.api.nvidia.com/v1'
    )
    
    print('\nSending request to LLM...')
    response = client.chat.completions.create(
        model='nvidia/nemotron-3-nano-30b-a3b',
        temperature=0.2,
        max_tokens=320,
        messages=[
            {"role": "system", "content": RISK_SYSTEM_PROMPT},
            {"role": "user", "content": test_prompt},
        ],
    )
    
    print('\n' + '=' * 70)
    print('RAW RESPONSE:')
    print('=' * 70)
    print(f'Response type: {type(response)}')
    print(f'Choices: {len(response.choices)}')
    
    if response.choices:
        content = response.choices[0].message.content
        print(f'\nContent type: {type(content)}')
        print(f'Content length: {len(content) if content else 0}')
        print(f'\nContent:\n{content}')
        
        # Try to parse as JSON
        print('\n' + '=' * 70)
        print('JSON PARSING:')
        print('=' * 70)
        
        if content:
            try:
                # Try direct parse first
                parsed = json.loads(content)
                print('✓ Direct JSON parse successful!')
                print(f'Parsed: {parsed}')
            except json.JSONDecodeError as e:
                print(f'✗ Direct JSON parse failed: {e}')
                
                # Try extracting JSON block
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_block = content[start:end+1]
                    print(f'\nExtracted JSON block: {json_block[:100]}...')
                    try:
                        parsed = json.loads(json_block)
                        print('✓ Extracted JSON parse successful!')
                        print(f'Parsed: {parsed}')
                    except json.JSONDecodeError as e2:
                        print(f'✗ Extracted JSON parse failed: {e2}')
        
except Exception as e:
    import traceback
    print(f'\n✗ ERROR: {e}')
    traceback.print_exc()
