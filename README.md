# Speculative Edits

## Problem
Your goal is to implement "speculative edits" using pytorch and huggingface with temperature 0 (greedy sampling)

### Speculative Edits

In speculative decoding, a draft model produces draft tokens that the large model verifies. Because the larger model can verify several tokens at once, it is faster
than generating tokens one at a time.

Instead of using a draft model to produce these draft tokens, we can produce the draft tokens ourselves. How? Most edits are just rewrites of a given region, with some minor modifications.

Consider the following prompt:
````txt
## Instructions
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

```ts
````

You should be able to generate this code much faster than vanilla token generation
with speculative edits. Why?

A sample generated response is:
````
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
  # This is a comment
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
````

On the first forward pass, we can feed in the entire original code block as a speculation. Then we start generating tokens when the model disagrees with our draft. (at the line `# this is a comment`). Eventually, we'll want to re-speculate on the remainder of the prompt.
Note that the choice of greedy sampling simplifies things.

If developing locally, we'd recommend using a small model like `gpt-2`.

If you would like to test real performance on smarter models before submission, you may use the card provided in the form to purchase $10 worth of GPU-hours. We'd recommend 3090s or 4090s on [Runpod](https://runpod.io), but any machine with >18GB of GPU RAM will suffice.

You may need to slightly modify the prompt/work with the instruction format of the model to get the desired behavior as well (of adding a single line comment, while rewriting the code block).

A good model to test would be `llama-3-8b-instruct` or `deepseek-coder-6.7b-instruct`.
## Submission Instructions
When finished, please zip this folder and upload the zipfile to this form: https://docs.google.com/forms/d/1COjBZBA5jSxpaLtBZlD-ktAS0LnWbiH4LfyUgzowgpg/edit
