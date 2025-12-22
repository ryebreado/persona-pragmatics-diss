from transformer_lens import HookedTransformer
import json

model = HookedTransformer.from_pretrained("gpt2-small")

with open("data/scalar_implicature_full.json") as f:
    examples = json.load(f)

def format_prompt(ex):
    return f"""{ex['scenario']}
{ex['items']}
{ex['outcome']}
{ex['question']}

Statement: "{ex['statement']}"

Is this statement an accurate answer to the question? Answer yes or no."""

for ex in examples[:5]:  # just first 5 to start
    prompt = format_prompt(ex)
    
    # Generate a few tokens
    output = model.generate(
        prompt, 
        max_new_tokens=10,
        temperature=0  # deterministic
    )
    
    print(f"Test {ex['test_id']} ({ex['category']})")
    print(f"Expected: {ex['expected']}")
    print(f"Model output: {output}")
    print("-" * 40)