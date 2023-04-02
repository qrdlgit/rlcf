
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
