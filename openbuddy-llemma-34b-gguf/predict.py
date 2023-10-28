#!/usr/bin/env python

import os
import time
import json

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import numpy as np
import pprint as pp
from cog import BasePredictor, Input, Path, BaseModel, ConcatenateIterator
from llama_cpp import Llama


MODEL_ID = "sprint-mammoth/openbuddy-mistral-7b-v13.1-GGUF"
CACHE_DIR = "checkpoints"

PROMPT_TEMPLATE = '''You are a helpful high school Math tutor. If you don't know the answer to a question, please don't share false information. You can speak fluently in many languages.
User: Hi
Assistant: Hello, how can I help you?</s>
User: {prompt}
Assistant:'''

DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_DO_SAMPLE = True
DEFAULT_NUM_BEAMS = 1
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
DEFAULT_PRESENCE_PENALTY = 1.1  # 1.15
DEFAULT_FREQUENCY_PENALTY = 0.2  # 0.2
DEFAULT_REPETITION_PENALTY = 1.0  # 1.0


class Predictor(BasePredictor):

    def setup(self):
        model_path = "/models/openbuddy-mistral-7b-v13.1-Q4_K_M.gguf"
        self.llm = Llama(
            model_path, n_ctx=4096, n_gpu_layers=-1, main_gpu=0, n_threads=1
        )

    def predict(
        self,
        prompt: str,
        max_new_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            ge=1,
            le=3500,
            default=DEFAULT_MAX_NEW_TOKENS,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.", default=DEFAULT_TEMPERATURE
        ),
        top_p: float = Input(
            description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            ge=0.01,
            le=1.0,
            default=DEFAULT_TOP_P,
        ),
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
            ge=1,
            le=100,
            default=DEFAULT_TOP_K,
        ),
        do_sample: bool = Input(
            description="Whether or not to use sampling ; use greedy decoding otherwise.",
            default=DEFAULT_DO_SAMPLE,
        ),
        num_beams: int = Input(
            description="Number of beams for beam search. 1 means no beam search.",
            ge=1,
            le=10,
            default=DEFAULT_NUM_BEAMS,
        ),
        repetition_penalty: float = Input(
            description="Repetition penalty, (float, *optional*, defaults to 1.0): The parameter for repetition penalty. 1.0 means no penalty. values greater than 1 discourage repetition, less than 1 encourage it. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.",
            ge=0.01,
            le=5,
            default=DEFAULT_REPETITION_PENALTY,
        ),
        presence_penalty: float = Input(
            description="Presence penalty, ",
            default=DEFAULT_PRESENCE_PENALTY,
        ),
        frequency_penalty: float = Input(
            description="Frequency penalty",
            default=DEFAULT_FREQUENCY_PENALTY,
        ),
        prompt_template: str = Input(
            description="The template used to format the prompt. The input prompt is inserted into the template using the `{prompt}` placeholder.",
            default=PROMPT_TEMPLATE,
        ),
        padding_mode: bool = Input(
            description="Whether to pad the left side of the prompt with eos token.",
            default=True,
        ),
        stop_sequences: str = Input(
            description="A comma-separated list of sequences to stop generation at. For example, '<end>,<stop>' will stop generation at the first instance of 'end' or '<stop>'.",
            default=None,
        ),
        debug: bool = Input(
            description="provide debugging output in logs",
            default=False,
        ),
    ) -> ConcatenateIterator:
        # Initialize prompt and generation config
        instruct_prompt = prompt_template.format(prompt=prompt)
        print(f"Your formatted prompt is: \n{instruct_prompt}")

        n_tokens = 0
        output = ""
        start = time.perf_counter()

        for tok in self.llm(
            instruct_prompt,
            max_tokens=max_new_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repetition_penalty,
            # mirostat_mode={"Disabled": 0, "Mirostat": 1, "Mirostat 2.0": 2}[mirostat_mode],
            # mirostat_eta=mirostat_learning_rate,
            # mirostat_tau=mirostat_entropy,
        ):
            text = tok["choices"][0]["text"]
            if text == "":
                continue
            yield text
            n_tokens += 1
            if n_tokens == 1 and debug:
                second_start = time.perf_counter()
                print(f"after initialization, first token took {second_start - start:.3f}")
            if debug:
                print(text)
            output += text
        endtime = time.perf_counter()
        duration = endtime - start
        print(f"Final output:{output}")
        print(f"\nGenerated in {duration} seconds.")

        if debug:
            print(f"Tokens per second: {n_tokens / duration:.2f}")
            print(f"Tokens per second not including time to first token: {(n_tokens -1) / (endtime - second_start):.2f}")
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")

        '''response = self.llm(
            instruct_prompt,
            max_tokens=max_new_tokens,
        )
        endtime = time.perf_counter()
        duration = endtime - start
        print(f"\nGenerated in {duration} seconds.")

        output = response["choices"][0]["text"]
        output = json.dumps(json.loads(output), indent=2)
        print(f"Final output:{output}")

        return output'''
        

        
    