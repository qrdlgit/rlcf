import argparse
from training_utils import (
    get_prompts_and_code_filenames,
    generate_code_improvements,
    update_generated_code_files,
    prepare_dataset,
    train_model,
)
from eval_rewards import (
    get_rewards_for_code_improvements
)

import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main(input_file):
    # Read input data
    prompt_and_code_files = get_prompts_and_code_filenames(input_file)
    code_files = [x[1] for x in prompt_and_code_files]
    
    # Define the model path and check if the fine-tuned model exists
    model_path = "codebert_finetuned"
    if os.path.exists(model_path):
        # Load the fine-tuned model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        # Load the pre-trained CodeBERT model and tokenizer
        model_name = "microsoft/CodeBERTa-small-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    
    # RLCF iterative process
    num_iterations = 5
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}")

        # Generate code improvements using the current model
        (improvement_prompt, code_file, generated_improvement) = generate_code_improvements(prompt_and_code_files[0], prompt_and_code_files[1], tokenizer, model)

        # Update generated code files with improvements
        update_generated_code_files(code_file, generated_improvement)

        # Evaluate the generated code improvements
        rewards = get_rewards_for_code_improvements([code_file])

        # Prepare the dataset for training
        dataset = prepare_dataset([improvement_prompt], [generated_code_improvement], rewards)

        # Train the model
        train_model(model, tokenizer, dataset)
        
    model.save_pretrained(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Improvement Trainer")
    parser.add_argument("input_file", help="Path to the input file containing array of tuples")
    args = parser.parse_args()
    main(args.input_file)
