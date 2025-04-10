from transformers import BertTokenizer, WhisperTokenizer
from typing import Any, List, Tuple


class PromptBuilder:
    def __init__(self, args: Any):
        self.args = args
        self.context = []

        if "bert" in self.args.model_type:
            self.tokenizer = BertTokenizer.from_pretrained(self.args.model_name)
            self.prompt_prefix_tokens = [self.tokenizer.cls_token]
            self.utterance_prefix_tokens = []
            self.utterance_suffix_tokens = [self.tokenizer.sep_token]
            self.padding_direction = "right"
        else:
            self.tokenizer = WhisperTokenizer.from_pretrained(self.args.model_name)
            self.prompt_prefix_tokens = ["<|startoftranscript|>", "<|notimestamps|>"]
            self.utterance_prefix_tokens = self.tokenizer.tokenize("Utterance -> ")
            self.utterance_suffix_tokens = self.tokenizer.tokenize("\n")
            self.padding_direction = "left"

        self.prompt_prefix_len = len(self.prompt_prefix_tokens)
        self.utterance_prefix_len = len(self.utterance_prefix_tokens)
        self.utterance_suffix_len = len(self.utterance_suffix_tokens)

    def reset_context(self):
        self.context = []

    def get_input_ids_and_attention_mask(self, utterance: str) -> Tuple[List[int], List[int]]:
        utterance_tokens = self.tokenizer.tokenize(utterance)
        if len(self.context) > self.args.context_size:
            self.context.pop(0)
        self.context.append(utterance_tokens)

        max_utterences_len = self.args.max_seq_length - self.prompt_prefix_len - self.utterance_prefix_len - self.utterance_suffix_len
        utterance_tokens = self.context[-1][-max_utterences_len:]
        utterances_tokens = self.utterance_prefix_tokens + utterance_tokens + self.utterance_suffix_tokens

        max_utterences_len = max_utterences_len - len(utterance_tokens)
        for previous_utterance_tokens in reversed(self.context[:-1]):
            previous_utterance_len = self.utterance_prefix_len + len(previous_utterance_tokens) + self.utterance_suffix_len
            if max_utterences_len - previous_utterance_len < 0:
                break
            utterances_tokens = self.utterance_prefix_tokens + previous_utterance_tokens + self.utterance_suffix_tokens + utterances_tokens
            max_utterences_len -= previous_utterance_len

        prompt_tokens = self.prompt_prefix_tokens + utterances_tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(prompt_tokens)
        attention_mask = [1] * len(input_ids)

        padding_length = self.args.max_seq_length - len(input_ids)
        padding_token_id = self.tokenizer.pad_token_id
        if self.padding_direction == "right":
            input_ids = input_ids + [padding_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        else:
            input_ids = [padding_token_id] * padding_length + input_ids
            attention_mask = [0] * padding_length + attention_mask

        return input_ids, attention_mask
