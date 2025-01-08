#!/bin/bash

API_URL="https://speculative-edits-production.up.railway.app"

echo "Testing Speculative Method..."
curl -X POST "$API_URL/edit" \
  -H "Content-Type: application/json" \
  -d '{"method": "speculative"}' | jq .

echo -e "\nTesting Vanilla Method..."
curl -X POST "$API_URL/edit" \
  -H "Content-Type: application/json" \
  -d '{"method": "vanilla"}' | jq .

echo -e "\nTesting with Custom Parameters..."
curl -X POST "$API_URL/edit" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "speculative",
    "max_tokens": 500
  }' | jq . 