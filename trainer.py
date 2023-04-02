from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from RestrictedPython import compile_restricted, safe_globals
import sys
from io import StringIO
import ast
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments


model_name = "microsoft/codebert-base"  # Replace with your fine-tuned model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def is_code_syntactically_correct(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def execute_code_safely(code, safe_globals):
    compiled_code = compile_restricted(code, "<inline>", "exec")
    exec(compiled_code, safe_globals)


def evaluate_performance(generated_code, input_data, expected_output, accuracy_threshold):
    if not is_code_syntactically_correct(generated_code):
        return False, "Code is not syntactically correct"

    safe_globals = {
        "__builtins__": safe_globals,
    }

    original_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        execute_code_safely(generated_code, safe_globals)
    except Exception as e:
        sys.stdout = original_stdout
        return False, f"Error during code execution: {str(e)}"

    sys.stdout = original_stdout

    if 'evaluate' not in safe_globals:
        return False, "The 'evaluate' function is not defined in the code"

    evaluate_func = safe_globals['evaluate']
    accuracy = evaluate_func(input_data, expected_output)

    if accuracy >= accuracy_threshold:
        return True, f"Accuracy: {accuracy}"
    else:
        return False, f"Accuracy: {accuracy} (below threshold)"


def reward_model(evaluation_results, weights):
    reward = 0
    reward += evaluation_results["ml_metric_improvement"] * weights["ml_metric_improvement"]
    return reward


def generate_code_improvement(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code


class CodeImprovementDataset(Dataset):
    def __init__(self, tokenized_prompts, tokenized_improvements, rewards):
        self.tokenized_prompts = tokenized_prompts
        self.tokenized_improvements = tokenized_improvements
        self.rewards = rewards

    def __len__(self):
        return len(self.tokenized_improvements)

    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenized_prompts[idx]['input_ids'],
            'attention_mask': self.tokenized_prompts[idx]['attention_mask'],
            'labels': self.tokenized_improvements[idx]['input_ids'],
            'reward': self.rewards[idx],
        }


def custom_compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    rewards = inputs.pop("reward")
    outputs = model(**inputs)
    logits = outputs.logits

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

    loss = loss * rewards.mean()

    return (loss, outputs) if return_outputs else loss


# RLCF iterative process
num_iterations = 5
for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}")

    prompts = [...]  # List of prompts used for generating the code improvements

    # Generate code improvements using the current model (use your generate_code_improvement function)
    generated_code_improvements = [generate_code_improvement(prompt) for prompt in prompts]

    input_data = [...]  # Your input data for testing the generated code
    expected_output = [...]  # Your expected output (ground truth) for the input data
    accuracy_threshold = 0.75

    # Evaluate the generated code improvements and get the rewards
    evaluation_results = [evaluate_performance(code_improvement, input_data, expected_output, accuracy_threshold) for code_improvement in generated_code_improvements]
    rewards = [reward_model(result, {"ml_metric_improvement": 50}) for success, result in evaluation_results if success]

    # Tokenize the prompts and the code improvements
    tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    tokenized_improvements = tokenizer(generated_code_improvements, return_tensors="pt", padding=True, truncation=True)

    # Create the custom dataset
    code_improvement_dataset = CodeImprovementDataset(tokenized_prompts, tokenized_improvements, rewards)

    # Set up the training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
    )

    # Train the model using the custom dataset and the training configuration
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=code_improvement_dataset,
        tokenizer=tokenizer,
        compute_loss=custom_compute_loss,
    )

    trainer.train()
