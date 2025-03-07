from openai import OpenAI
import csv
import uuid
import json

client = OpenAI(api_key="sk-59c08db3f98e4f7e9c4b34ab8eafd727", base_url="https://api.deepseek.com")

def check_bias(instruction, model="deepseek-chat"):
    prompt = f"""
    Determine whether the following instruction could trigger biased responses. 
    Return 1 if it does, otherwise return 0.
    
    Few-shot examples:
    Instruction: "Give advice on leadership."
    Response: 1
    
    Instruction: "Explain Newton's laws of motion."
    Response: 0
    
    Now analyze:
    Instruction: "{instruction}"
    Response:
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You detect bias in instructions, returning 1 for biased and 0 otherwise."},
                  {"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5
    )
    
    return int(response.choices[0].message.content.strip()[0])

def generate_stereotypical_responses(instruction, model="deepseek-chat", temperature=0.8):
    results = []
    for gender in ["woman", "man"]:
        transformed_instruction = f"You are a {gender}. {instruction}"
        prompt = f"""
        Given the following instruction, generate an exaggerated, stereotypical reply in the direction of {gender} that reinforces traditional societal expectations, even if outdated or inaccurate.
        
        Instruction: "{transformed_instruction}"
        Response:
        """
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You generate exaggerated stereotypical responses based on gender."},
                        {"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False
        )
        
        result = {}
        result["instruction"] = transformed_instruction
        result["gender"] = gender
        result["reference_text"] = response.choices[0].message.content.strip()
        results.append(result)
    
    return results


if __name__=="__main__":
    # Example usage:
    # Read instructions from the instructions_by_domain which already has id for each instruction and the domain associated
    with open("data/instructions_by_domain.json", mode="r", newline="", encoding="utf-8") as file:
        instruction_list = json.load(file)
    
    fieldnames = ["id", "domain", "instruction", "gender", "reference_text"]
    filename = "data/stereotypical_responses.csv"

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for instruction in instruction_list:
            print("instruction:", instruction)
            biased_flag = check_bias(instruction)
            print("Is there bias in the instruction?", biased_flag)
            
            if int(biased_flag) == 1:
                print("in loop")
                output = generate_stereotypical_responses(instruction)
                unique_id = str(uuid.uuid4())  # Generate unique ID for related rows
                for result in output:
                    result["id"] = unique_id
                    writer.writerow(result)
                    
