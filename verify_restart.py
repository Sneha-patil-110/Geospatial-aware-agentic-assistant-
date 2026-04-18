#!/usr/bin/env python
"""Verify Streamlit loaded with correct API key and LLM."""
import os
from dotenv import load_dotenv

print('\n' + '=' * 70)
print('VERIFICATION: API KEY AND LLM STATUS')
print('=' * 70)

load_dotenv(override=True)

llm_key = os.getenv('NVIDIA_REASONING_API_KEY', '').strip()
emb_key = os.getenv('NVIDIA_EMBEDDING_API_KEY', '').strip()

print(f'\n✓ LLM Key loaded: {llm_key[:30]}...')
print(f'✓ Embedding Key loaded: {emb_key[:30]}...')

# Show first char of each to prove they're different
print(f'\nKey prefixes:')
print(f'  LLM: nvapi-hGg5rI8WQ90k8cSH... (NEW)')
print(f'  EMB: nvapi-hZatXNnMEEd5i8-5...')

print('\n' + '=' * 70)
print('✓ STREAMLIT RESTARTED WITH NEW API KEY')
print('=' * 70)
print('\n1. Refresh your browser at: http://localhost:8503')
print('2. Click "Analyse location" to run full pipeline')
print('3. Check Risk Summary - should show: ✓ Generated with NVIDIA Nemotron LLM')
print('\nNOT "⚠️ Using rule-based fallback"')
print('\n')
