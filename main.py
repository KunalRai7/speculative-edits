"""You should implement both of these methods.

Vanilla edits should just be a custom generate loop with a huggingface
transformer.

Speculative edits should implement the speculative editing algorithm.

To test these, make sure they work on the prompt provided in the README"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional

def get_readme_prompt() -> str:
    """Returns the example prompt from README."""
    return '''## Instructions
Rewrite the code and add a single line comment above `const [shouldRefreshGold, setShouldRefreshGold]`...

## Code

```ts
export default function Visualization() {
  const [instanceIdInputs, setInstanceIdInputs] = createSignal<
    InstanceId[] | null
  >(null);
  const [storedInput, setStoredInput] = createSignal<string>("");
  const [datapointOptions, setDatapointOptions] = createSignal<PropsInstance[]>(
    []
  );
  const [shouldRefreshGold, setShouldRefreshGold] =
    createSignal<boolean>(false);
  const [showGold, setShowGold] = createSignal<boolean>(false);
  const [selectedGoldRequestId, setSelectedGoldRequestId] = createSignal<
    string | undefined
  >(undefined);
  const [goldInstances, setGoldInstances] = createSignal<
    {
      sessionId: string;
      email: string | undefined;
      requestId: string | undefined;
      dateAdded: Date;
      type: $Enums.CppGoldExampleType;
    }[]
  >([]);
}
```

## Rewritten code

```ts'''

def speculative_edit(prompt: str, max_tokens: int) -> str:
    """Implements speculative decoding for faster text generation.
    First feeds the entire original code block as speculation,
    then generates when model disagrees (expected at comment insertion).
    Finally re-speculates on the remainder.
    
    Args:
        prompt: Input text to be edited
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated/edited text
    """
    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    def get_next_token_probs(input_ids: torch.Tensor) -> torch.Tensor:
        """Get probability distribution for next token."""
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            return torch.softmax(next_token_logits, dim=-1)
    
    # Extract code blocks and sections
    code_start = prompt.find("```ts\n") + 6
    code_end = prompt.find("```\n\n## Rewritten")
    original_code = prompt[code_start:code_end]
    
    # Split prompt into parts
    prefix = prompt[:code_start]  # Everything before code block
    code_block = original_code    # The code block to use as speculation
    suffix = prompt[code_end:]    # Everything after code block
    
    # Tokenize all parts
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    code_ids = tokenizer(code_block, return_tensors="pt").input_ids.to(device)
    
    # Start with prefix
    output_ids = prefix_ids
    tokens_generated = len(prefix_ids[0])
    
    # Try to use entire code block as speculation first
    i = 0
    while i < len(code_ids[0]) and tokens_generated < max_tokens:
        probs = get_next_token_probs(output_ids)
        predicted_token = torch.argmax(probs, dim=-1)
        
        if predicted_token == code_ids[0][i]:
            # Model agrees with original code
            output_ids = torch.cat([output_ids, predicted_token.unsqueeze(0).unsqueeze(0)], dim=1)
            tokens_generated += 1
            i += 1
        else:
            # Model disagrees - this should happen when it wants to insert the comment
            # Generate tokens normally until we potentially get back on track
            divergence_point = i
            generation_buffer = []
            
            while tokens_generated < max_tokens:
                probs = get_next_token_probs(output_ids)
                next_token = torch.argmax(probs, dim=-1).unsqueeze(0).unsqueeze(0)
                output_ids = torch.cat([output_ids, next_token], dim=1)
                tokens_generated += 1
                generation_buffer.append(next_token)
                
                # Try to find a point where we can resume speculation
                # Look for a sequence of matching tokens in the remaining code
                remaining_tokens = code_ids[0][divergence_point:]
                buffer_tensor = torch.cat(generation_buffer, dim=1)
                
                # Look for matches of increasing length in the buffer
                for match_length in range(min(len(generation_buffer), 3), 0, -1):
                    if len(remaining_tokens) >= match_length:
                        buffer_seq = buffer_tensor[0, -match_length:]
                        for j in range(len(remaining_tokens) - match_length + 1):
                            if torch.all(buffer_seq == remaining_tokens[j:j+match_length]):
                                # Found a match, resume speculation from here
                                i = divergence_point + j + match_length
                                break
                        if i != divergence_point:  # If we found a match
                            break
                
                if i != divergence_point:  # If we found a match
                    break
                
                # If we've generated too much without finding a match, stop looking
                if len(generation_buffer) > 50:  # Arbitrary threshold
                    i = len(code_ids[0])  # Skip to end of speculation
                    break
    
    # Generate any remaining tokens
    while tokens_generated < max_tokens:
        probs = get_next_token_probs(output_ids)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(0).unsqueeze(0)
        output_ids = torch.cat([output_ids, next_token], dim=1)
        tokens_generated += 1
        
        # Check for completion
        if tokenizer.eos_token_id in output_ids[0]:
            break
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def vanilla_edit(prompt: str, max_tokens: int) -> str:
    """Standard token-by-token generation using Hugging Face transformers.
    Implements a custom generation loop as required.
    
    Args:
        prompt: Input text to be edited
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated/edited text
    """
    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Custom generation loop with temperature=0 (greedy)
    tokens_generated = 0
    output_ids = input_ids
    
    while tokens_generated < max_tokens:
        # Get next token probabilities
        with torch.no_grad():
            outputs = model(output_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
        
        # Greedy selection (temperature=0)
        next_token = torch.argmax(next_token_probs, dim=-1).unsqueeze(0).unsqueeze(0)
        
        # Append next token to output
        output_ids = torch.cat([output_ids, next_token], dim=1)
        tokens_generated += 1
        
        # Check for completion
        if tokenizer.eos_token_id in output_ids[0]:
            break
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def test_implementations():
    """Test both implementations with the README prompt."""
    prompt = get_readme_prompt()
    max_tokens = 1000
    
    print("Testing vanilla edit...")
    vanilla_result = vanilla_edit(prompt, max_tokens)
    print("\nVanilla result:")
    print(vanilla_result)
    
    print("\nTesting speculative edit...")
    spec_result = speculative_edit(prompt, max_tokens)
    print("\nSpeculative result:")
    print(spec_result)

if __name__ == "__main__":
    test_implementations()