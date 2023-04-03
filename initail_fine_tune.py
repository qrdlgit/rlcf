import torch
from transformers import CodeBertTokenizer, CodeBertForConditionalGeneration
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load the pre-trained model
model_name = "microsoft/codebert-base"
tokenizer = CodeBertTokenizer.from_pretrained(model_name)

model = CodeBertForConditionalGeneration.from_pretrained(model_name)

# Create a custom loss function
def custom_loss(outputs, labels, tokenizer, loss_fct=CrossEntropyLoss()):
    find_code_token = tokenizer.encode("find code:", add_special_tokens=False)
    replace_code_token = tokenizer.encode("replace code:", add_special_tokens=False)
    mask = torch.zeros_like(labels)

    for i in range(labels.size(0)):
        for j in range(labels.size(1) - len(find_code_token)):
            if torch.equal(labels[i, j : j + len(find_code_token)], torch.tensor(find_code_token)):
                mask[i, j : j + len(find_code_token)] = 1
            if torch.equal(labels[i, j : j + len(replace_code_token)], torch.tensor(replace_code_token)):
                mask[i, j : j + len(replace_code_token)] = 1

    lprobs = torch.nn.functional.log_softmax(outputs.view(-1, outputs.size(-1)), dim=-1)
    active_loss = mask.view(-1) == 1
    active_lprobs = lprobs.view(-1, outputs.size(-1))[active_loss]
    active_labels = labels.view(-1)[active_loss]
    loss = loss_fct(active_lprobs, active_labels)
    return loss
  
# Modify the Trainer class to use the custom loss function
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        mask = (labels != -100).float()
        outputs = model(**inputs)
        loss = custom_loss(outputs.logits, labels, mask)
        return (loss, outputs) if return_outputs else loss
      
# Prepare the dataset
def create_dataset(file_path, tokenizer, block_size=128, separator="--"):
    with open(file_path, "r") as f:
        content = f.read()

    examples = content.split(separator)
    encoded_examples = []

    for example in examples:
        encoded_example = tokenizer.encode(example, add_special_tokens=True, max_length=block_size, truncation=True)
        encoded_examples.append(encoded_example)

    dataset = torch.utils.data.TensorDataset(torch.tensor(encoded_examples, dtype=torch.long))
    return dataset

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

# Prepare the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Train the model with the custom trainer
trainer = CustomTrainer(
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
