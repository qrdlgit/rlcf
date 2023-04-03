from RestrictedPython import compile_restricted, safe_globals
import sys
from io import StringIO
import ast
import pylint.lint
import pycodestyle
from io import StringIO
import sys

def is_code_syntactically_correct(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def execute_code_safely(code, safe_globals):
    compiled_code = compile_restricted(code, "<inline>", "exec")
    exec(compiled_code, safe_globals)


def count_pylint_warnings_errors(code):
    pylint_output = StringIO()
    sys.stdout = pylint_output
    pylint.lint.Run([code], do_exit=False)
    sys.stdout = sys.__stdout__

    warning_count = 0
    error_count = 0

    for line in pylint_output.getvalue().split("\n"):
        if line.startswith("W:"):
            warning_count += 1
        elif line.startswith("E:"):
            error_count += 1

    return warning_count, error_count


def count_pycodestyle_errors(code):
    style_guide = pycodestyle.StyleGuide()
    result = style_guide.check_files([code])
    return result.total_errors


def get_evaluation_results(code_filename):
    with open(code_filename, "r") as f:
        code = f.read()

    evaluation_results = {
        "compiles": is_code_syntactically_correct(code),
        "ml_metric_improvement": 0,  # To be replaced by the actual ML metric improvement
        "number_of_code_warnings": 0,
        "number_of_code_syntax_errors": 0,
        "number_of_style_errors": 0,
    }

    if evaluation_results["compiles"]:
        # Execute the code and check for the existence of an 'evaluate' function
        safe_globals = {"__builtins__": safe_globals}
        execute_code_safely(code, safe_globals)

        if "evaluate" in safe_globals:
            evaluate_func = safe_globals["evaluate"]

            # You need to provide your input_data and expected_output here
            input_data = [...]  # Your input data for testing the generated code
            expected_output = [...]  # Your expected output (ground truth) for the input data

            evaluation_results["ml_metric_improvement"] = evaluate_func(input_data, expected_output)

    # Count warnings and errors in the code using pylint
    warning_count, error_count = count_pylint_warnings_errors(code_filename)
    evaluation_results["number_of_code_warnings"] = warning_count
    evaluation_results["number_of_code_syntax_errors"] = error_count

    # Count style errors in the code using pycodestyle
    evaluation_results["number_of_style_errors"] = count_pycodestyle_errors(code_filename)

    return evaluation_results


def reward_model(evaluation_results, weights):
    reward = 0

    # Check if the code compiles
    if evaluation_results["compiles"]:
        reward += weights["compiles"]

        # ML metric improvement
        reward += evaluation_results["ml_metric_improvement"] * weights["ml_metric_improvement"]

    # Deduct rewards based on the number of errors and warnings
    reward -= evaluation_results["number_of_code_warnings"] * weights["number_of_code_warnings"]
    reward -= evaluation_results["number_of_code_syntax_errors"] * weights["number_of_code_syntax_errors"]
    reward -= evaluation_results["number_of_style_errors"] * weights["number_of_style_errors"]

    return reward
      
   
def calculate_rewards(evaluation_results):
    # Define the weights for different evaluation results
    weights = {
        "compiles": 10,
        "ml_metric_improvement": 50,
        "number_of_code_warnings": 1,
        "number_of_code_syntax_errors": 5,
        "number_of_style_errors": 1,
    }

    rewards = reward_model(evaluation_results, weights)
    return rewards
  
def get_rewards_for_code_improvements(code_files):
    rewards = []

    for code_file in code_files:
        # Get the evaluation results for the code file
        evaluation_results = get_evaluation_results(code_file)

        # Calculate the reward based on the evaluation results
        reward = calculate_reward(evaluation_results)

        # Append the reward to the rewards list
        rewards.append(reward)

    return rewards


