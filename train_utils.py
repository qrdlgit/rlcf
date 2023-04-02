
import json

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

    return (code_file, generated_improvement)

