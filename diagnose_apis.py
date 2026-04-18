#!/usr/bin/env python
"""Diagnostic check for Safety Zone Combined LLM setup."""
import os
import sys
from dotenv import load_dotenv

print('\n' + '=' * 70)
print('SAFETY ZONE COMBINED - API KEY DIAGNOSTIC')
print('=' * 70)

# Load environment
load_dotenv(override=True)

# Check embedding key
emb_key = os.getenv('NVIDIA_EMBEDDING_API_KEY', '').strip()
llm_key = os.getenv('NVIDIA_REASONING_API_KEY', '').strip()

print('\n[1/3] Environment Variables:')
print(f'  NVIDIA_EMBEDDING_API_KEY: {"✓ LOADED" if emb_key else "✗ MISSING"}')
print(f'  NVIDIA_REASONING_API_KEY: {"✓ LOADED" if llm_key else "✗ MISSING"}')

if not emb_key or not llm_key:
    print('\n✗ ERROR: API keys not found in .env file!')
    print('   Make sure .env file has both keys set.')
    sys.exit(1)

# Test embedding API
print('\n[2/3] Testing Embedding API...')
try:
    from openai import OpenAI
    client = OpenAI(
        api_key=emb_key,
        base_url='https://integrate.api.nvidia.com/v1'
    )
    response = client.embeddings.create(
        input=['test'],
        model='nvidia/nv-embedqa-e5-v5',
        encoding_format='float',
        extra_body={'input_type': 'query', 'truncate': 'NONE'}
    )
    print(f'  ✓ Embedding API works (vector dim: {len(response.data[0].embedding)})')
except Exception as e:
    print(f'  ✗ Embedding API failed: {str(e)[:80]}')
    sys.exit(1)

# Test LLM API
print('\n[3/3] Testing LLM API...')
try:
    client = OpenAI(
        api_key=llm_key,
        base_url='https://integrate.api.nvidia.com/v1'
    )
    response = client.chat.completions.create(
        model='nvidia/nemotron-3-nano-30b-a3b',
        messages=[{'role': 'user', 'content': 'Say OK'}],
        temperature=0.2,
        max_tokens=10
    )
    content = response.choices[0].message.content
    print(f'  ✓ LLM API works (response: "{content}")')
except Exception as e:
    print(f'  ✗ LLM API failed: {str(e)[:80]}')
    sys.exit(1)

print('\n' + '=' * 70)
print('✓✓✓ ALL SYSTEMS READY - START/RESTART STREAMLIT APP ✓✓✓')
print('=' * 70)
print('\nRun: streamlit run streamlit_app.py')
print('\n')
