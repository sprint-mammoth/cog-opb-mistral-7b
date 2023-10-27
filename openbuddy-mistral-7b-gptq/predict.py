import os
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from cog import BasePredictor, Input, ConcatenateIterator
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer
from threading import Thread


MODEL_ID = "TheBloke/openbuddy-mistral-7B-v13-GPTQ"
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
        # model_name_or_path = "TheBloke/openbuddy-mistral-7B-v13-GPTQ"
        # To use a different branch, change revision
        # For example: revision="gptq-4bit-32g-actorder_True"
        self.lm_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_ID,
            device_map="auto",
            cache_dir="../openbuddy/",
            trust_remote_code=False,
            revision="main"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=MODEL_ID,
            use_fast=True, 
            cache_dir="../openbuddy/"
        )
        self.text_streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, Timeout=5, skip_special_tokens=True)

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
        prompt = prompt_template.format(prompt=prompt)
        print(f"Your formatted prompt is: \n{prompt}")

        if stop_sequences:
            stop_sequences = stop_sequences.split(",")

        if padding_mode:
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
        '''
        *greedy decoding* by calling [~generation.GenerationMixin.greedy_search] if num_beams=1 and do_sample=False
        *contrastive search* by calling [~generation.GenerationMixin.contrastive_search] if penalty_alpha>0. and top_k>1
        *multinomial sampling* by calling [~generation.GenerationMixin.sample] if num_beams=1 and do_sample=True
        *beam-search decoding* by calling [~generation.GenerationMixin.beam_search] if num_beams>1 and do_sample=False
        *beam-search multinomial sampling* by calling [~generation.GenerationMixin.beam_sample] if num_beams>1 and do_sample=True
        *diverse beam-search decoding* by calling [~generation.GenerationMixin.group_beam_search], if num_beams>1 and num_beam_groups>1
        *constrained beam-search decoding* by calling [~generation.GenerationMixin.constrained_beam_search], if constraints!=None or force_words_ids!=None
        *assisted decoding* by calling [~generation.GenerationMixin.assisted_decoding], if assistant_model is passed to .generate()
        '''
        generation_config = GenerationConfig(
            eos_token_id=self.lm_model.config.eos_token_id,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty, # repetition_penalty (float, *optional*, defaults to 1.0): The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        )
        # generation_config.save_pretrained("openbuddy", "user_generation_config.json") # generate config could be personalized and saved
        n_tokens = 0
        start = time.time()
        model_inputs = self.tokenizer([prompt_template], padding=True, return_tensors="pt").to("cuda")

        # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        generation_kwargs = dict(model_inputs, streamer=self.text_streamer, generation_config=generation_config)
        thread = Thread(target=self.lm_model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        for new_text in self.text_streamer:
            n_tokens += 1
            yield new_text
            if n_tokens == 1 and debug:
                second_start = time.time()
                print(f"after initialization, first token took {second_start - start:.3f}")
            if debug:
                print(new_text) # for debug
            generated_text += new_text
        endtime = time.time()
        duration = endtime - start
        print(f"Final output:{generated_text}")
        print(f"\nGenerated in {duration} seconds.")

        if debug:
            print(f"Tokens per second: {n_tokens / duration:.2f}")
            print(f"Tokens per second not including time to first token: {(n_tokens -1) / (endtime - second_start):.2f}")
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")
    