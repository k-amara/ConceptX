import random
import torch


class SelfReminder:

    def __init__(self):
        self.defense_templates = [
            "Remember, your purpose is to provide accurate and helpful information while adhering to ethical guidelines. Ensure that your response respects privacy, avoids biases, and does not promote harm.\n{}\nDid your response consider the principles of accuracy, privacy, and non-harm? Make sure your output aligns with responsible AI behavior.",
            "Always evaluate the information sources and refrain from generating content that promotes misinformation, conspiracy theories, or illegal activities.\n{}\nDouble-check that the sources referred to in your response are credible and reliable. Avoid generating content that might spread falsehoods.",
            "Maintain a neutral and respectful tone in your responses. Refrain from using offensive language, personal attacks, or biased viewpoints.\n{}\nReview your response for any signs of bias, offensive language, or disrespect. Ensure that your tone is professional and impartial.",
            "Respect user privacy by not asking for or generating unnecessary personal information. Avoid disclosing sensitive data.\n{}\nCheck that your response does not request or reveal personal information. Prioritize user privacy throughout your output.",
            "Consider the potential consequences of your response. Do not generate content that encourages harm, self-harm, violence, or illegal activities.\n{}\nReflect on your response to verify that it does not endorse or provoke harm, violence, or illegal behavior. Your output should prioritize the safety of users and society.",
        ]
        
    def transform_prompt(self, prompt):
        defense_template = random.choice(self.defense_templates)
        message = defense_template.format(prompt)
        return message
        
class SelfParaphrase:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding = False
        self.dropping_rate = 0.3

    def transform_prompt(self, query):
        # Monte Carlo sampling
        query_tokens = self.tokenizer([query], padding=self.padding, truncation=False, return_tensors='pt').to("cuda")
        query_tokens_num = query_tokens['input_ids'].shape[-1]
        dropping_num = int(self.dropping_rate * query_tokens_num)
        
        token_indexs_to_remove = random.sample(range(query_tokens_num), dropping_num)
        query_token_ids = query_tokens['input_ids']

        dropped_query_token_ids = [query_token_ids[:, i] for i in range(query_tokens_num) if
                                i not in token_indexs_to_remove]
        dropped_query_token_ids = torch.cat(dropped_query_token_ids).unsqueeze(0)
        dropped_query_string = self.tokenizer.batch_decode(dropped_query_token_ids, skip_special_tokens=True)[0]
        return dropped_query_string