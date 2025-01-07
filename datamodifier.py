import pandas as pd
from openai import OpenAI
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

def load_file(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)

def parse_instruction(instruction, data):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a data manipulation assistant. Generate Python code using pandas to modify the given DataFrame."},
            {"role": "user", "content": f"""Here is a data task: '{instruction}'. 
            The dataset has the following structure:
            {data.head().to_string()}

            Generate the Python code to perform this operation. 
            The DataFrame is named 'data'.
            Modify the 'data' DataFrame directly. Do not create new DataFrames unless necessary.
            If you create a new DataFrame, assign it back to 'data'.
            Do not include any print statements or DataFrame creation code.
            Only include the code necessary to modify the existing 'data' DataFrame.
            Do not include any Markdown formatting or code block indicators."""}
        ],
        model="gpt-4o",
        temperature=0.7,
        max_tokens=4096,
        top_p=1
    )
    return response.choices[0].message.content.strip()

def modify_data(data, instruction):
    
    generated_code = parse_instruction(instruction, data)
    print("Generated Code:\n", generated_code)   
    cleaned_code = generated_code.strip('`').replace('python\n', '', 1)
    
    try:
        local_vars = {"data": data.copy()}  
        exec(cleaned_code, {"pd": pd}, local_vars)  

        if 'data' in local_vars and isinstance(local_vars['data'], pd.DataFrame):
            modified_df = local_vars['data']
        else:
            
            dataframes = [v for v in local_vars.values() if isinstance(v, pd.DataFrame)]
            if dataframes:
                modified_df = dataframes[0]  
            else:
                print("Warning: The generated code did not produce a valid DataFrame. Returning original data.")
                return data

        print("\nModified DataFrame:")
        print(modified_df)  
        return modified_df  
    except Exception as e:
        print(f"Error during execution of the generated code: {e}")
        print("Error details:")
        import traceback
        traceback.print_exc()
        return data

def log_changes(log_file, change_details):
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()}, {change_details}\n")

def save_file(data, output_path):
    if output_path.endswith('.csv'):
        data.to_csv(output_path, index=False)
    elif output_path.endswith('.xlsx'):
        data.to_excel(output_path, index=False)
    else:
        raise ValueError("Unsupported output format. Please use CSV or Excel.")

def main():
    file_path = input("Enter the file path: ")
    instruction = input("Enter your modification instruction: ")
    output_path = input("Enter the output file path: ")
    log_file = "modification_log.txt"

    try:
        data = load_file(file_path)
        modified_data = modify_data(data, instruction)
        
        save_file(modified_data, output_path)
        log_changes(log_file, f"Instruction: {instruction}")
        print("File modified and saved successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
