from transformers import WhisperTokenizer
from typing import Any

PROMPT_UTTERANCE_PREFIX = "Utterance -> "
PROMPT_PREVIOUS_UTTERANCES_PREFIX = """Previous Utterances:
"""
PROMPT_LAST_UTTERANCE_PREFIX = f"""Last Utterance:
{PROMPT_UTTERANCE_PREFIX}"""


class PromptBuilderWithoutTranscript:
    def __init__(self, args: Any, tokenizer: WhisperTokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.prompt_tokens = ["<|startoftranscript|>", "<|notimestamps|>"]
    
    def get_prompt_tokens(self) -> list[str]:
        return self.prompt_tokens


class PromptBuilderWithTranscript:
    def __init__(self, args: Any, tokenizer: WhisperTokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.prompt_prefix_tokens = ["<|startoftranscript|>", "<|notimestamps|>"]
        self.prompt_prefix_length = len(self.prompt_prefix_tokens)

        self.utterance_prefix_tokens = tokenizer.tokenize(PROMPT_UTTERANCE_PREFIX)
        self.utterance_prefix_length = len(self.utterance_prefix_tokens)

        self.prompt_previous_utterances_prefix_tokens = tokenizer.tokenize(PROMPT_PREVIOUS_UTTERANCES_PREFIX)
        self.prompt_previous_utterances_prefix_length = len(self.prompt_previous_utterances_prefix_tokens)

        self.prompt_last_utterance_prefix_tokens = tokenizer.tokenize(PROMPT_LAST_UTTERANCE_PREFIX)
        self.prompt_last_utterance_prefix_length = len(self.prompt_last_utterance_prefix_tokens)

    def get_prompt_tokens(self, previous_utterance_tokens_list: list[list[str]], utterance_tokens: list[str]) -> list[str]:
        max_utterance_len = self.args.max_seq_length - self.prompt_prefix_length - self.prompt_last_utterance_prefix_length
        utterance_tokens = utterance_tokens[-max_utterance_len:]
        last_utterance_tokens = self.prompt_last_utterance_prefix_tokens + utterance_tokens

        previous_utterances_tokens = []
        if len(previous_utterance_tokens_list) > 0:
            max_previous_utterences_len = max_utterance_len - len(utterance_tokens) - self.prompt_previous_utterances_prefix_length
            if max_previous_utterences_len > 0:
                for previous_utterance_tokens in reversed(previous_utterance_tokens_list):
                    max_previous_utterences_len -= self.utterance_prefix_length + len(previous_utterance_tokens)
                    if max_previous_utterences_len < 0:
                        break
                    previous_utterances_tokens = self.utterance_prefix_tokens + previous_utterance_tokens + previous_utterances_tokens
                if len(previous_utterances_tokens) > 0:
                    previous_utterances_tokens = self.prompt_previous_utterances_prefix_tokens + previous_utterances_tokens

        tokens = self.prompt_prefix_tokens + previous_utterances_tokens + last_utterance_tokens
        return tokens 