import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TextDataset, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForMaskedLM

from transformers import DataCollatorForLanguageModeling

# Load the pre-trained model
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

def custom_mlm_loss(outputs, labels, tokenizer, loss_fct=CrossEntropyLoss(), focus_weight=5.0):
    find_code_token = tokenizer.encode("find code:", add_special_tokens=False)
    replace_code_token = tokenizer.encode("replace code:", add_special_tokens=False)
    mask = torch.zeros_like(labels)

    # Create a mask for the specific tokens
    for i in range(labels.size(0)):
        for j in range(labels.size(1) - len(find_code_token)):
            if torch.equal(labels[i, j : j + len(find_code_token)], torch.tensor(find_code_token)):
                mask[i, j : j + len(find_code_token)] = 1
            if torch.equal(labels[i, j : j + len(replace_code_token)], torch.tensor(replace_code_token)):
                mask[i, j : j + len(replace_code_token)] = 1

    # MLM loss
    lprobs = torch.nn.functional.log_softmax(outputs.view(-1, outputs.size(-1)), dim=-1)
    active_loss = mask.view(-1) == 1
    inactive_loss = mask.view(-1) == 0
    masked_labels = labels.view(-1)
    
    # Compute loss for the specific tokens
    active_lprobs = lprobs.view(-1, outputs.size(-1))[active_loss]
    active_labels = masked_labels[active_loss]
    active_loss = loss_fct(active_lprobs, active_labels)

    # Compute loss for other tokens
    inactive_lprobs = lprobs.view(-1, outputs.size(-1))[inactive_loss]
    inactive_labels = masked_labels[inactive_loss]
    inactive_loss = loss_fct(inactive_lprobs, inactive_labels)

    # Combine the losses, applying a weight to the specific tokens
    loss = focus_weight * active_loss + inactive_loss
    return loss



# Modify the Trainer class to use the custom loss function
class CustomTrainer(Trainer):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = custom_mlm_loss(outputs.logits, labels, self.tokenizer)
        return (loss, outputs) if return_outputs else loss

def create_dataset(file_path, tokenizer, block_size=128, separator="--"):
    with open(file_path, "r") as f:
        content = f.read()

    examples = content.split(separator)
    encoded_examples = []

    for example in examples:
        encoded_example = tokenizer.encode(example, add_special_tokens=True, max_length=block_size, truncation=True, padding="max_length")
        encoded_examples.append(torch.tensor(encoded_example, dtype=torch.long))

    return encoded_examples

train_dataset = create_dataset("train.txt", tokenizer)
eval_dataset = create_dataset("eval.txt", tokenizer)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_codebert",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=50,
    save_steps=50,
    warmup_steps=50,
    evaluation_strategy="steps",
    logging_dir="./logs",
)


# Custom data collator
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def _torch_collate_batch(self, examples, tokenizer):
        # Call the parent class method
        return super()._torch_collate_batch(examples, tokenizer)
    


data_collator = CustomDataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# Train the model with the custom trainer
trainer = CustomTrainer(
    tokenizer=tokenizer,
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("./codebert_finetuned")
tokenizer.save_pretrained("./codebert_finetuned")
