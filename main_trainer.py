import argparse
from training_utils import (
    get_prompts_and_code_filenames,
    generate_code_improvements,
    update_generated_code_files,
    evaluate_code_improvements,
    calculate_rewards,
    prepare_dataset,
    train_model,
)

def main(input_file):
    # Read input data
    prompt_and_code_files = get_prompts_and_code_filenames(input_file)

    # RLCF iterative process
    num_iterations = 5
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}")

        # Generate code improvements using the current model
        generated_code_improvements = generate_code_improvements(prompt_and_code_files)

        # Update generated code files with improvements
        update_generated_code_files(generated_code_improvements, prompt_and_code_files)

        # Evaluate the generated code improvements
        evaluation_results = evaluate_code_improvements(generated_code_improvements, prompt_and_code_files)

        # Calculate rewards based on evaluation results
        rewards = calculate_rewards(evaluation_results)

        # Prepare the dataset for training
        dataset = prepare_dataset(prompt_and_code_files, generated_code_improvements, rewards)

        # Train the model
        train_model(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Improvement Trainer")
    parser.add_argument("input_file", help="Path to the input file containing array of tuples")
    args = parser.parse_args()
    main(args.input_file)
