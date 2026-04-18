#!/usr/bin/env python
"""Quick test of APIs only."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print('QUICK API VALIDATION TEST')
print('=' * 70)

# Test 1: Embedding
print('\n[1/2] Testing Embedding API...')
try:
    client = OpenAI(
        api_key=os.getenv('NVIDIA_EMBEDDING_API_KEY'),
        base_url='https://integrate.api.nvidia.com/v1'
    )
    response = client.embeddings.create(
        input=['Forest fires in India'],
        model='nvidia/nv-embedqa-e5-v5',
        encoding_format='float',
        extra_body={'input_type': 'query', 'truncate': 'NONE'}
    )
    dim = len(response.data[0].embedding)
    print(f'✓ EMBEDDING API WORKS')
    print(f'  • Model: nvidia/nv-embedqa-e5-v5')
    print(f'  • Vector dimension: {dim}')
    print(f'  • Query: "Forest fires in India"')
except Exception as e:
    print(f'✗ EMBEDDING API FAILED: {e}')
    exit(1)

# Test 2: LLM
print('\n[2/2] Testing LLM API...')
try:
    client = OpenAI(
        api_key=os.getenv('NVIDIA_REASONING_API_KEY'),
        base_url='https://integrate.api.nvidia.com/v1'
    )
    response = client.chat.completions.create(
        model='nvidia/nemotron-3-nano-30b-a3b',
        messages=[{'role': 'user', 'content': 'Briefly define forest fire in one sentence.'}],
        temperature=0.2,
        max_tokens=50
    )
    content = response.choices[0].message.content
    print(f'✓ LLM API WORKS')
    print(f'  • Model: nvidia/nemotron-3-nano-30b-a3b')
    print(f'  • Response: "{content}"')
except Exception as e:
    print(f'✗ LLM API FAILED: {e}')
    exit(1)

print('\n' + '=' * 70)
print('✓✓✓ BOTH APIs ARE WORKING - ALL SYSTEMS GO ✓✓✓')
print('=' * 70)
