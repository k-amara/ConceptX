# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-59c08db3f98e4f7e9c4b34ab8eafd727", base_url="https://api.deepseek.com")


def create_prompt_for_replacement(sentence, input_concepts):             
    prompt = f"""
        You are an AI assistant that neutralizes concepts in sentences. Your task is to replace given concepts with neutral alternatives that neutralize their semantic importance while preserving grammatical correctness. The replacements must NOT be synonyms or somehow close in meaning.

        Example Input:
        "sentence": "Describe the ideal qualities of a leader in a team.",
        "input_concepts": ["Describe", "qualities", "leader", "team"]
        Example Output:
        "replacements": ["Mention", "aspects", "individual", "group"]

        Given the following sentence and concepts:

        Sentence: "{sentence}"
        Concepts: {input_concepts}

        For each concept, replace it with a new word that:
        - Neutralizes its semantic importance. This will strongly weaken their semantic importance in the sentence.
        - Preserves grammatical correctness.
        - Is NOT a synonym or somehow close in meaning.

        Return only a Python list of concepts in this format:
        ["neutralized_concept_1", "neutralized_concept_2", "neutralized_concept_3", ...]
        Please do not include any additional explanation, sentences, or content other than the list.
        """
    return prompt

def get_multiple_completions(prompt, model="deepseek-chat", num_sequences=3, temperature=1.0):
    responses = []
    
    for _ in range(num_sequences):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
                ],
            temperature=temperature,
            stream=False
        )
        responses.append(response.choices[0].message.content)
    
    return responses



if __name__=="__main__":
    sentence = "The artist created a stunning masterpiece that captivated the audience."
    input_concepts = ["artist", "stunning", "masterpiece", "captivated", "audience"]
    
    # Example usage
    prompt = create_prompt_for_replacement(sentence, input_concepts)
    completions = get_multiple_completions(prompt, num_sequences=3)

    # Print all k generated completions
    for i, completion in enumerate(completions):
        print(f"Completion {i+1}: {completion}\n")