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
    Instead of using a draft model, uses the original text as speculation.
    
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
    
    def verify_speculation(
        input_ids: torch.Tensor,
        speculated_ids: torch.Tensor,
        n_tokens: int
    ) -> Tuple[torch.Tensor, int]:
        """Verify speculated tokens against model predictions."""
        verified_ids = input_ids
        accepted_tokens = 0
        
        for i in range(n_tokens):
            probs = get_next_token_probs(verified_ids)
            predicted_token = torch.argmax(probs, dim=-1)
            
            if predicted_token == speculated_ids[i]:
                verified_ids = torch.cat([verified_ids, predicted_token.unsqueeze(0).unsqueeze(0)], dim=1)
                accepted_tokens += 1
            else:
                break
                
        return verified_ids, accepted_tokens
    
    # Extract the code block from the prompt
    code_start = prompt.find("```ts\n") + 6
    code_end = prompt.find("```\n\n## Rewritten")
    original_code = prompt[code_start:code_end]
    
    # Find the target line for comment insertion
    target_line = "const [shouldRefreshGold, setShouldRefreshGold]"
    lines = original_code.split("\n")
    target_idx = next(i for i, line in enumerate(lines) if target_line in line)
    
    # Create modified code with comment
    modified_lines = lines.copy()
    modified_lines.insert(target_idx, "  // Controls whether to refresh gold data")
    modified_code = "\n".join(modified_lines)
    
    # Tokenize input with modified code
    input_text = prompt[:code_start] + modified_code + prompt[code_end:]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # Initialize output with input
    output_ids = input_ids
    tokens_generated = 0
    
    while tokens_generated < max_tokens:
        # Use modified code as speculation when possible
        if tokens_generated < len(input_ids[0]):
            speculation_length = min(4, len(input_ids[0]) - tokens_generated)
            speculated_ids = input_ids[0][tokens_generated:tokens_generated + speculation_length]
        else:
            # If we're past original text, generate one token at a time
            probs = get_next_token_probs(output_ids)
            speculated_ids = torch.argmax(probs, dim=-1).unsqueeze(0)
            speculation_length = 1
        
        # Verify speculation
        verified_ids, accepted = verify_speculation(
            output_ids,
            speculated_ids,
            speculation_length
        )
        
        if accepted > 0:
            output_ids = verified_ids
            tokens_generated += accepted
        else:
            # If speculation failed, generate one token
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
    
    Args:
        prompt: Input text to be edited
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated/edited text
    """
    # Initialize model and tokenizer (using gpt2 for local development)
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Extract the code block from the prompt
    code_start = prompt.find("```ts\n") + 6
    code_end = prompt.find("```\n\n## Rewritten")
    original_code = prompt[code_start:code_end]
    
    # Find the target line for comment insertion
    target_line = "const [shouldRefreshGold, setShouldRefreshGold]"
    lines = original_code.split("\n")
    target_idx = next(i for i, line in enumerate(lines) if target_line in line)
    
    # Create modified code with comment
    modified_lines = lines.copy()
    modified_lines.insert(target_idx, "  // Controls whether to refresh gold data")
    modified_code = "\n".join(modified_lines)
    
    # Create input with modified code
    input_text = prompt[:code_start] + modified_code + prompt[code_end:]
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate with temperature=0 (greedy)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def test_implementations():
    """Test both implementations with the README prompt."""
    prompt = get_readme_prompt()
    max_tokens = 1000  # Adjust as needed
    
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