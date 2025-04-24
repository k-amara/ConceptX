#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)"""
import os
import inspect
import logging
from typing import Tuple
import time

import torch
from accelerate import PartialState
from accelerate.utils import set_seed
from dotenv import load_dotenv

import time
import random
from functools import wraps
from openai import RateLimitError

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BloomForCausalLM,
    BloomTokenizerFast,
    CTRLLMHeadModel,
    CTRLTokenizer,
    GenerationMixin,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2TokenizerFast,
    GPTJForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    OPTForCausalLM,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from huggingface_hub import login

from openai import OpenAI, BadRequestError
from deepseek_tokenizer import ds_token as DSTokenizer


def safe_login():
    marker_file = os.path.expanduser("~/.hf_logged_in")

    if not os.path.exists(marker_file):
        load_dotenv()
        login(token=os.getenv("HF_TOKEN"))

        # Create a marker file so other jobs know login happened
        with open(marker_file, "w") as f:
            f.write("logged_in")

safe_login()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
    "gptj": (GPTJForCausalLM, AutoTokenizer),
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "opt": (OPTForCausalLM, GPT2Tokenizer),
    "mistral-7b-it": (AutoModelForCausalLM, AutoTokenizer)
}

MODEL_IDENTIFIER = {
    "gpt2": "gpt2",
    "xlnet": "xlnet-base-cased",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-3-3b": "meta-llama/Llama-3.2-3B",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-3-4b": "google/gemma-3-4b-it",
    "mistral-7b-it": "mistralai/Mistral-7B-Instruct-v0.2"
}

# Set quantization configuration based on the user's input
quantization_map = {
    "8bit": BitsAndBytesConfig(load_in_8bit=True),
    "4bit": BitsAndBytesConfig(load_in_4bit=True)
}

def retry_on_ratelimit(max_retries=10, base_delay=1.0, backoff_factor=2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    wait = delay + random.uniform(0, 0.5)
                    print(f"[RateLimit] Attempt {attempt + 1}/{max_retries}, retrying in {wait:.2f} seconds...")
                    time.sleep(wait)
                    delay *= backoff_factor
            raise RateLimitError("Max retries exceeded due to rate limiting.")
        return wrapper
    return decorator


# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in MODEL_IDENTIFIER[args.model]
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def sparse_model_config(model_config):
    embedding_size = None
    if hasattr(model_config, "hidden_size"):
        embedding_size = model_config.hidden_size
    elif hasattr(model_config, "n_embed"):
        embedding_size = model_config.n_embed
    elif hasattr(model_config, "n_embd"):
        embedding_size = model_config.n_embd

    num_head = None
    if hasattr(model_config, "num_attention_heads"):
        num_head = model_config.num_attention_heads
    elif hasattr(model_config, "n_head"):
        num_head = model_config.n_head

    if embedding_size is None or num_head is None or num_head == 0:
        raise ValueError("Check the model config")

    num_embedding_size_per_head = int(embedding_size / num_head)
    if hasattr(model_config, "n_layer"):
        num_layer = model_config.n_layer
    elif hasattr(model_config, "num_hidden_layers"):
        num_layer = model_config.num_hidden_layers
    else:
        raise ValueError("Number of hidden layers couldn't be determined from the model config")

    return num_layer, num_head, num_embedding_size_per_head


def generate_past_key_values(model, batch_size, seq_len):
    num_block_layers, num_attention_heads, num_embedding_size_per_head = sparse_model_config(model.config)
    if model.config.model_name == "bloom":
        past_key_values = tuple(
            (
                torch.empty(int(num_attention_heads * batch_size), num_embedding_size_per_head, seq_len)
                .to(model.dtype)
                .to(model.device),
                torch.empty(int(num_attention_heads * batch_size), seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
            )
            for _ in range(num_block_layers)
        )
    else:
        past_key_values = tuple(
            (
                torch.empty(batch_size, num_attention_heads, seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
                torch.empty(batch_size, num_attention_heads, seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
            )
            for _ in range(num_block_layers)
        )
    return past_key_values


def prepare_jit_inputs(inputs, model, tokenizer):
    batch_size = len(inputs)
    dummy_input = tokenizer.batch_encode_plus(inputs, return_tensors="pt")
    dummy_input = dummy_input.to(model.device)
    if model.config.use_cache:
        dummy_input["past_key_values"] = generate_past_key_values(model, batch_size, 1)
    dummy_input["attention_mask"] = torch.cat(
        [
            torch.zeros(dummy_input["attention_mask"].shape[0], 1)
            .to(dummy_input["attention_mask"].dtype)
            .to(model.device),
            dummy_input["attention_mask"],
        ],
        -1,
    )
    return dummy_input


class _ModelFallbackWrapper(GenerationMixin):
    __slots__ = ("_optimized", "_default")

    def __init__(self, optimized, default):
        self._optimized = optimized
        self._default = default

    def __call__(self, *args, **kwargs):
        if kwargs["past_key_values"] is None and self._default.config.use_cache:
            kwargs["past_key_values"] = generate_past_key_values(self._default, kwargs["input_ids"].shape[0], 0)
        kwargs.pop("position_ids", None)
        for k in list(kwargs.keys()):
            if kwargs[k] is None or isinstance(kwargs[k], bool):
                kwargs.pop(k)
        outputs = self._optimized(**kwargs)
        lm_logits = outputs[0]
        past_key_values = outputs[1]
        fixed_output = CausalLMOutputWithPast(
            loss=None,
            logits=lm_logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
        return fixed_output

    def __getattr__(self, item):
        return getattr(self._default, item)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs
    ):
        return self._default.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs
        )

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return self._default._reorder_cache(past_key_values, beam_idx)



class LLMPipeline:
    def __init__(self, args):
        self.args = args
        
        # Initialize the distributed state.
        self.distributed_state = PartialState(cpu=self.args.use_cpu)

        logger.warning(f"device: {self.distributed_state.device}, 16-bits inference: {args.fp16}")

        if args.seed is not None:
            set_seed(args.seed)


        self.args.model_name = args.model_name.lower()
        self.dataset_name = args.dataset

        # Check if the model is part of the Llama series
        if any(model in self.args.model_name.lower() for model in ["llama", "gemma"]):
            model_id = MODEL_IDENTIFIER[self.args.model_name]
            self.pipe = pipeline(
                "text-generation", 
                model=model_id, 
                tokenizer=model_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            self.model = self.pipe.model
            self.tokenizer = self.pipe.tokenizer
            self.is_llama = True
        else:
            self.is_llama = False
            # Initialize the model and tokenizer
            try:
                model_class, tokenizer_class = MODEL_CLASSES[self.args.model_name]
            except KeyError:
                raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

            self.tokenizer = tokenizer_class.from_pretrained(MODEL_IDENTIFIER[self.args.model_name])
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Select the quantization configuration based on the argument
            quantization_config = quantization_map.get(args.quantization, None)

            # Load the model with the appropriate configuration
            if quantization_config:
                self.model = model_class.from_pretrained(
                    MODEL_IDENTIFIER[self.args.model_name], 
                    attn_implementation="sdpa", 
                    quantization_config=quantization_config
                )
            else:
                self.model = model_class.from_pretrained(
                    MODEL_IDENTIFIER[self.args.model_name], 
                    attn_implementation="sdpa",
                    torch_dtype=torch.bfloat16,  
                    device_map="auto"  
                )
            
            
            self.model = torch.compile(self.model)
            # Set the model to the right device
            self.model.to(self.distributed_state.device)

            if args.fp16:
                self.model.half()
            max_seq_length = getattr(self.model.config, "max_position_embeddings", 0)
            self.args.length = adjust_length_to_model(args.length, max_sequence_length=max_seq_length)
            logger.info(args)
                
            if args.jit:
                jit_input_texts = ["enable jit"]
                jit_inputs = prepare_jit_inputs(jit_input_texts, self.model, self.tokenizer)
                torch._C._jit_set_texpr_fuser_enabled(False)
                self.model.config.return_dict = False
                if hasattr(self.model, "forward"):
                    sig = inspect.signature(self.model.forward)
                else:
                    sig = inspect.signature(self.model.__call__)
                jit_inputs = tuple(jit_inputs[key] for key in sig.parameters if jit_inputs.get(key, None) is not None)
                traced_model = torch.jit.trace(self.model, jit_inputs, strict=False)
                traced_model = torch.jit.freeze(traced_model.eval())
                traced_model(*jit_inputs)
                traced_model(*jit_inputs)

                self.model = _ModelFallbackWrapper(traced_model, self.model)
            
            
    def generate(self, input_text):
        prompt = create_prompt(input_text, self.dataset_name, api_required=False)
        prompt_text = prompt if prompt else input("Model prompt >>> ")
        
        if self.is_llama:
            # Use the pipeline directly for Llama models
            outputs = self.pipe(
                prompt_text, 
                max_new_tokens=self.args.length, 
                temperature=self.args.temperature,
                top_k=self.args.k,
                top_p=self.args.p,
                repetition_penalty=self.args.repetition_penalty,
                do_sample=self.args.do_sample,
                num_return_sequences=self.args.num_return_sequences,
                return_full_text=False
            )
            response = outputs[0]["generated_text"]

        else:
            # Different models need different input formatting and/or extra arguments
            requires_preprocessing = self.args.model_name in PREPROCESSING_FUNCTIONS.keys()
            if requires_preprocessing:
                prepare_input = PREPROCESSING_FUNCTIONS.get(self.args.model_name)
                preprocessed_prompt_text = prepare_input(self.args, self.model, self.tokenizer, prompt_text)

                if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                    tokenizer_kwargs = {"add_space_before_punct_symbol": True}
                else:
                    tokenizer_kwargs = {}

                encoded_prompt = self.tokenizer.encode(
                    preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
                )
            else:
                prefix = self.args.prefix if self.args.prefix else self.args.padding_text
                encoded_prompt = self.tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
            encoded_prompt = encoded_prompt.to(self.distributed_state.device)

            if encoded_prompt.size()[-1] == 0:
                input_ids = None
            else:
                input_ids = encoded_prompt

            attention_mask = torch.ones_like(input_ids)

            with torch.inference_mode():
                output_sequences = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=self.args.length + len(encoded_prompt[0]),
                        temperature=self.args.temperature,
                        top_k=self.args.k,
                        top_p=self.args.p,
                        repetition_penalty=self.args.repetition_penalty,
                        do_sample=self.args.do_sample,
                        num_return_sequences=self.args.num_return_sequences,
                    )

            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            generated_sequences = []
            generated_responses = [] # We add responses only without repeting the prompt

            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                # Remove all text after the stop token
                text = text[: text.find(self.args.stop_token) if self.args.stop_token else None]
                # Remove the excess text that was used for pre-processing
                post_text = text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
                # Add the prompt at the beginning of the sequence. 
                total_sequence = (
                    prompt_text + post_text
                )

                generated_sequences.append(total_sequence)
                generated_responses.append(post_text)
            response = generated_responses[0]
        return response # Only return the first resopnse if num_sequenes>1


MODEL_API_CLASSES = {
    "gpt4o-mini": ("GPT4O_MINI_API_KEY", "azure/gpt-4o-mini", (GPT2TokenizerFast,"Xenova/gpt-4o"), "https://aikey-gateway.ivia.ch"),
    "gpt4o": ("GPT4O_API_KEY", "azure/gpt-4o", (GPT2TokenizerFast,"Xenova/gpt-4o"), "https://aikey-gateway.ivia.ch"),
    "o1": ("O1_API_KEY", "azure/o1", (GPT2TokenizerFast,"Xenova/gpt-4o"), "https://aikey-gateway.ivia.ch"),
    "deepseek": ("DEEPSEEK_API_KEY", "deepseek-chat", DSTokenizer, "https://api.deepseek.com"),
}


class ContentPolicyViolationError(Exception):
    """Raised when the API request violates the content policy and is blocked."""
    pass

class LLMAPI:
    RATE_LIMIT = 100  # Max requests per minute

    def __init__(self, args, rate_limit_enabled=False):
        self.args = args
        self.args.model_name = args.model_name.lower()
        self.api_key, self.model_id, tokenizer, self.base_url = MODEL_API_CLASSES[self.args.model_name]
        
        self.client = OpenAI(
            api_key=os.getenv(self.api_key), 
            base_url=self.base_url 
        )

        if isinstance(tokenizer, tuple) and len(tokenizer) == 2:
            tokenizer = tokenizer[0].from_pretrained(tokenizer[1])
        self.tokenizer = tokenizer

        self.rate_limit_enabled = rate_limit_enabled  # Toggle rate limiting
        self.request_times = []
        self.dataset_name = args.dataset
        
        
    @retry_on_ratelimit()
    def generate(self, input_text):
        if self.rate_limit_enabled:
            self._enforce_rate_limit()
            
        prompt = create_prompt(input_text, self.dataset_name, api_required=True)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.args.temperature,
                stream=False
            )
            
            return response.choices[0].message.content

        except BadRequestError as e:
            error_message = str(e)
            if "ResponsibleAIPolicyViolation" in error_message or "ContentPolicyViolationError" in error_message:
                raise ContentPolicyViolationError("Azure OpenAI blocked the request due to content policy violation.")
            else:
                raise  # Re-raise other errors

    def _enforce_rate_limit(self):
        """Ensure we do not exceed 100 requests per minute."""
        now = time.time()
        self.request_times.append(now)

        # Remove timestamps older than 60 seconds
        self.request_times = [t for t in self.request_times if now - t < 60]

        if len(self.request_times) > self.RATE_LIMIT:
            wait_time = 60 - (now - self.request_times[0])
            time.sleep(wait_time)  # Delay to stay within rate limit
            
            
            
def process_instructions(df, llm):
    valid_ids = []
    valid_responses = []
    for _, row in df.iterrows():
        try:
            response = llm.generate(row['instruction'])
            valid_ids.append(row['id'])
            valid_responses.append(response)
        except ContentPolicyViolationError:
            continue  # Skip instructions that raise the error
    
    # Filter the DataFrame to keep only valid rows
    df_valid = df[df['id'].isin(valid_ids)].copy()
    df_valid['response'] = valid_responses  # Add the valid responses
    
    return df_valid


def create_prompt(text, dataset, api_required):
    if dataset in ["sentiment", "sst2"]:
        prompt = f"""Determine the sentiment of the following sentence: {text}. Your response must be either "positive" or "negative"."""
    elif dataset in ["alpaca", "genderbias", "genderbias2", "saladbench"]:
        if api_required:
            prompt = f"""
                Given the following instruction, provide an answer as direct advice. Do not use bullet points.
                Instruction: "{text}"
                Response:
                """
        else:
            prompt = f"""
                Given the following instruction, provide an answer as direct advice. Use continuous text. 
                Instruction: "{text}"
                Response:
                """
    else:
        raise  ("Unknown dataset type passed!")
    return prompt
        