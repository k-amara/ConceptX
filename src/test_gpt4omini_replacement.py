# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv
from model import ContentPolicyViolationError
# Load the .env file
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GPT4O_MINI_API_KEY"), # OpenAI API Key
    base_url="https://aikey-gateway.ivia.ch" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)


def create_prompt(word, sentence):             
    prompt = f"""
        Suggest an antonym word to replace the word "{word}" in the sentence {sentence}.
        Return only the new word as Python string.
        Please do not include any additional explanation, sentences, or content other than the list. """
    return prompt


def replace_to_harmless(prompt, model="azure/gpt-4o-mini", temperature=1.0):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            stream=False
        )
        response = response.choices[0].message.content
    
    except BadRequestError as e:
        error_message = str(e)
        if "ResponsibleAIPolicyViolation" in error_message or "ContentPolicyViolationError" in error_message:
            raise ContentPolicyViolationError("Azure OpenAI blocked the request due to content policy violation.")
        else:
            raise  # Re-raise other errors
            
    return response


sentence = "Forging a person’s signature or committing identity theft"
word = "theft"
prompt = create_prompt(word, sentence)
new_word = replace_to_harmless(prompt)
print("New word: ", new_word)