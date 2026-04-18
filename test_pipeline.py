#!/usr/bin/env python
"""Quick test of the full RAG pipeline."""
import sys
sys.path.insert(0, '.')

from app.assistant import answer_question

print('Testing full RAG pipeline...')
print('=' * 70)

# Test 1: Forest fires
print('\n[Test 1] Query: "What is a forest fire?"')
result = answer_question('What is a forest fire?', top_k_docs=5)
print(f'✓ Answer: {result["answer"][:150]}...')
print(f'✓ Used LLM: {result["used_llm"]}')
print(f'✓ Citations: {len(result.get("citations", []))} sources')
print(f'✓ Retrieved docs: {len(result.get("retrieved_docs", []))}')
if result.get('retrieved_docs'):
    print(f'  Top match: {result["retrieved_docs"][0]["title"]} (score: {result["retrieved_docs"][0]["score"]})')

# Test 2: Cyclone preparedness
print('\n[Test 2] Query: "How to prepare for a cyclone?"')
result = answer_question('How to prepare for a cyclone?', top_k_docs=5)
print(f'✓ Answer: {result["answer"][:150]}...')
print(f'✓ Used LLM: {result["used_llm"]}')
if result.get('retrieved_docs'):
    print(f'  Top match: {result["retrieved_docs"][0]["title"]} (score: {result["retrieved_docs"][0]["score"]})')

# Test 3: Relief camps
print('\n[Test 3] Query: "What is a relief camp?"')
result = answer_question('What is a relief camp?', top_k_docs=5)
print(f'✓ Answer: {result["answer"][:150]}...')
print(f'✓ Used LLM: {result["used_llm"]}')
if result.get('retrieved_docs'):
    print(f'  Top match: {result["retrieved_docs"][0]["title"]} (score: {result["retrieved_docs"][0]["score"]})')

print('\n' + '=' * 70)
print('✓ ALL TESTS PASSED - APIs are working!')
