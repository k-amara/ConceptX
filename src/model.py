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

import torch
from accelerate import PartialState
from accelerate.utils import set_seed
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

from transformers import (
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    CTRLLMHeadModel,
    CTRLTokenizer,
    GenerationMixin,
    GPT2LMHeadModel,
    GPT2Tokenizer,
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
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from huggingface_hub import login

# Step 1: Log in to Hugging Face using your access token
login(token=os.getenv("HF_TOKEN"))


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
}

MODEL_IDENTIFIER = {
    "gpt2": "gpt2",
    "xlnet": "xlnet-base-cased",
    "llama": "meta-llama/Llama-3.2-3B-Instruct"
}

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



class Model:
    def __init__(self, args):
        self.args = args
        
        # Initialize the distributed state.
        self.distributed_state = PartialState(cpu=self.args.use_cpu)

        logger.warning(f"device: {self.distributed_state.device}, 16-bits inference: {args.fp16}")

        if args.seed is not None:
            set_seed(args.seed)

        # Initialize the model and tokenizer
        try:
            args.model_name = args.model_name.lower()
            model_class, tokenizer_class = MODEL_CLASSES[args.model_name]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        self.tokenizer = tokenizer_class.from_pretrained(MODEL_IDENTIFIER[args.model_name])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = model_class.from_pretrained(MODEL_IDENTIFIER[args.model_name], attn_implementation="sdpa")
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
            
    def generate(self, prompt):
        prompt_text = prompt if prompt else input("Model prompt >>> ")

        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = self.args.model_name in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_name)
            preprocessed_prompt_text = prepare_input(args, self.model, self.tokenizer, prompt_text)

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
                    do_sample=True,
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
            print(total_sequence)

        return generated_responses[0] # Only return the first resopnse if num_sequenes>1


