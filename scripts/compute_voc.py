import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from tqdm import tqdm
import numpy as np

# Hyperparameter: cost per token
GAMMA = 0.1

# Load model + tokenizer (same as used for generation)
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load generations
with open("outputs/generated.json", "r") as f:
    examples = json.load(f)

scored = []

def get_logprob(prompt, target):
    """Compute log-probability of target given prompt"""
    full_input = prompt + target
    inputs = tokenizer(full_input, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_likelihood = -outputs.loss.item() * inputs["input_ids"].shape[1]
    return log_likelihood

for ex in tqdm(examples):
    q = ex["question"]
    y = ex["gold"]
    z = ex["cot"]  # chain of thought

    # Log probs
    logprob_direct = get_logprob(f"Q: {q}\nA: ", y)
    logprob_cot = get_logprob(f"Q: {q}\nLet's think step by step.\nA: {z}\nA: ", y)

    # Utility = increase in likelihood
    utility = logprob_cot - logprob_direct

    # Cost = length of reasoning (in tokens)
    cot_tokens = tokenizer.encode(z)
    cost = GAMMA * len(cot_tokens)

    voc = utility - cost

    scored.append({
        "id": ex["id"],
        "question": q,
        "gold": y,
        "direct": ex["direct"],
        "cot": z,
        "logprob_direct": round(logprob_direct, 3),
        "logprob_cot": round(logprob_cot, 3),
        "utility": round(utility, 3),
        "cost": round(cost, 3),
        "voc": round(voc, 3),
        "strategy": "cot" if voc > 0 else "direct"
    })

# Save results
with open("outputs/voc_scored.json", "w") as f:
    json.dump(scored, f, indent=2)

print("✅ VOC scoring complete! Saved to outputs/voc_scored.json")
