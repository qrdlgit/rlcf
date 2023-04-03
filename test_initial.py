import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def generate_prediction(prompt_file, model_path, tokenizer_path):
    # Load the fine-tuned model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Read the prompt file
    with open(prompt_file, "r") as f:
        prompt = f.read()

    # Tokenize the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Get the model's output logits
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits

    # Get the most probable tokens
    predicted_tokens = torch.argmax(logits, dim=-1)

    # Decode the tokens into text
    predicted_text = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)

    return predicted_text

# Test the model with a prompt file
prompt_file = "prompt.txt"
model_path = "./codebert_finetuned"
tokenizer_path = "./codebert_finetuned"

predicted_text = generate_prediction(prompt_file, model_path, tokenizer_path)
print(predicted_text)
