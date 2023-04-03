import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_text(prompt_filename, model, tokenizer, max_new_tokens=50):
    with open(prompt_filename, "r") as f:
        prompt = f.read().strip()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + max_new_tokens, num_return_sequences=1)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_text.py <prompt_filename>")
        sys.exit(1)

    prompt_filename = sys.argv[1]

    # Load the fine-tuned model and tokenizer
    model_path = "./codebert_finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    generated_text = generate_text(prompt_filename, model, tokenizer)
    print("Generated text:\n", generated_text)
