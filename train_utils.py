
import json
from transformers import (
    GPTNeoForCausalLM,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

# Update the update_generated_code_files function

def update_generated_code_files(generated_code_file, improvements):
    start_token = "<embed>"
    end_token = "</embed>"

    start_index = improvements.find(start_token) + len(start_token)
    end_index = improvements.find(end_token)

    improvements_json = improvements[start_index:end_index]
    improvements_list = json.loads(improvements_json)

    with open(generated_code_file, "r") as f:
        code = f.read()

    for improvement in improvements_list:
        find = improvement["find"]
        replace = improvement["replace"]
        code = code.replace(find, replace)

    with open(generated_code_file, "w") as f:
        f.write(code)

def get_prompt_for_code_improvement(prompt_file, code_file):
    with open(prompt_file, "r") as f:
        prompt = f.read()

    with open(code_file, "r") as f:
        code = f.read()

    improvement_prompt = f"{code}\n{prompt}"
    return improvement_prompt


def generate_code_improvements(prompt_file, code_file, tokenizer, model):
    improvement_prompt = get_prompt_for_code_improvement(prompt_file, code_file)

    inputs = tokenizer.encode(improvement_prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    generated_improvement = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return (improvement_prompt, code_file, generated_improvement)


from datasets import Dataset

class CodeImprovementDataset(Dataset):
    def __init__(self, prompts, responses, rewards):
        self.data = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            response = responses[i]
            reward = rewards[i]

            self.data.append({
                "prompt": prompt,
                "response": response,
                "reward": reward
            })

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def prepare_dataset(prompts, responses, rewards):
    return CodeImprovementDataset(prompts, responses, rewards)



def train_model(model, tokenizer, dataset, epochs=1, batch_size=8, learning_rate=5e-5, weight_decay=0.01, warmup_steps=0):
    # Create a DataLoader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(data_loader) * epochs)

    # Fine-tune the model
    model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            prompts = batch["prompt"]
            responses = batch["response"]
            rewards = batch["reward"]

            # Encode the input (prompts) and target (responses) texts
            input_encoding = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            target_encoding = tokenizer(responses, return_tensors="pt", padding=True, truncation=True)

            # Calculate the model's loss based on the rewards
            outputs = model(**input_encoding, labels=target_encoding.input_ids)
            loss = outputs.loss
            weighted_loss = loss / rewards

            # Backward pass and optimization step
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            scheduler.step()

