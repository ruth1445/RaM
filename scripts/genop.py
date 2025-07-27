import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from pathlib import Path
from tqdm import tqdm

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.eval()

# Make sure we're on CPU or CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load dataset
with open("data/data/sample.json", "r") as f:
    questions = json.load(f)

outputs = []

for ex in tqdm(questions):
    qid = ex["id"]
    question = ex["question"]
    gold = ex["answer"]

    # Direct answer prompt
    direct_prompt = f"Q: {question}\nA:"
    direct_inputs = tokenizer.encode(direct_prompt, return_tensors="pt").to(device)
    direct_output = model.generate(
        direct_inputs, max_length=direct_inputs.shape[1] + 10, do_sample=False
    )
    direct_text = tokenizer.decode(direct_output[0], skip_special_tokens=True)
    direct_answer = direct_text.split("A:")[-1].strip()

    # Chain-of-Thought (CoT) prompt
    cot_prompt = f"Q: {question}\nLet's think step by step.\nA:"
    cot_inputs = tokenizer.encode(cot_prompt, return_tensors="pt").to(device)
    cot_output = model.generate(
        cot_inputs,
        max_length=cot_inputs.shape[1] + 40,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    cot_text = tokenizer.decode(cot_output[0], skip_special_tokens=True)
    cot_answer = cot_text.split("A:")[-1].strip()

    outputs.append({
        "id": qid,
        "question": question,
        "gold": gold,
        "direct": direct_answer,
        "cot": cot_answer
    })

# Save results
Path("outputs").mkdir(exist_ok=True)
with open("outputs/generated.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("✅ Done. Saved to outputs/generated.json")
