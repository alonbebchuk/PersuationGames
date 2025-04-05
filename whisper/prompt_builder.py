from transformers import WhisperTokenizer
from typing import Any

PROMPT_PREFIX = """Task:
Determine whether the last utterance falls under the {strategy} strategy.

Strategy Definition:
{strategy} - {strategy_definition}.

"""
PROMPT_UTTERANCE_PREFIX = "Utterance -> "
PROMPT_PREVIOUS_UTTERANCES_PREFIX = """Previous Utterances:
"""
PROMPT_LAST_UTTERANCE_PREFIX = f"""Last Utterance:
{PROMPT_UTTERANCE_PREFIX}"""
PROMPT_SUFFIX = """
Strategy: {strategy}

Does the last utterance fall under this strategy?
Answer: """
STRATEGY_TO_STRATEGY_DEFINITION = {
    "Identity Declaration": "State one's own role or identity in the game",
    "Accusation": "Claim someone has a specific identity or strategic behavior",
    "Interrogation": "Questions about someone's identity or behavior",
    "Call for Action": "Encourage people to take an action during the game",
    "Defense": "Defending yourself or someone else against an accusation or defending a game-related argument",
    "Evidence": "Provide a body of game-related facts or information",
}


class PromptBuilderWithoutTranscript:
    def __init__(self, args: Any, tokenizer: WhisperTokenizer):
        self.args = args
        self.tokenizer = tokenizer

        prompt_prefix = PROMPT_PREFIX.format(strategy=self.args.strategy, strategy_definition=STRATEGY_TO_STRATEGY_DEFINITION[self.args.strategy])
        prompt_suffix = PROMPT_SUFFIX.format(strategy=self.args.strategy)
        self.prompt_tokens = ["<|startoftranscript|>", "<|notimestamps|>"] + tokenizer.tokenize(prompt_prefix) + tokenizer.tokenize(prompt_suffix)
    
    def get_prompt_tokens(self) -> list[str]:
        return self.prompt_tokens


class PromptBuilderWithTranscript:
    def __init__(self, args: Any, tokenizer: WhisperTokenizer):
        self.args = args
        self.tokenizer = tokenizer

        prompt_prefix = PROMPT_PREFIX.format(strategy=self.args.strategy, strategy_definition=STRATEGY_TO_STRATEGY_DEFINITION[self.args.strategy])
        self.prompt_prefix_tokens = ["<|startoftranscript|>", "<|notimestamps|>"] + tokenizer.tokenize(prompt_prefix)
        self.prompt_prefix_length = len(self.prompt_prefix_tokens)

        self.utterance_prefix_tokens = tokenizer.tokenize(PROMPT_UTTERANCE_PREFIX)
        self.utterance_prefix_length = len(self.utterance_prefix_tokens)

        self.prompt_previous_utterances_prefix_tokens = tokenizer.tokenize(PROMPT_PREVIOUS_UTTERANCES_PREFIX)
        self.prompt_previous_utterances_prefix_length = len(self.prompt_previous_utterances_prefix_tokens)

        self.prompt_last_utterance_prefix_tokens = tokenizer.tokenize(PROMPT_LAST_UTTERANCE_PREFIX)
        self.prompt_last_utterance_prefix_length = len(self.prompt_last_utterance_prefix_tokens)

        prompt_suffix = PROMPT_SUFFIX.format(strategy=self.args.strategy)
        self.prompt_suffix_tokens = tokenizer.tokenize(prompt_suffix)
        self.prompt_suffix_length = len(self.prompt_suffix_tokens)

    def get_prompt_tokens(self, previous_utterance_tokens_list: list[list[str]], utterance_tokens: list[str]) -> list[str]:
        max_utterance_len = self.args.max_seq_length - \
            self.prompt_prefix_length - \
            self.prompt_last_utterance_prefix_length - \
            self.prompt_suffix_length
        utterance_tokens = utterance_tokens[-max_utterance_len:]
        last_utterance_tokens = self.prompt_last_utterance_prefix_tokens + utterance_tokens

        previous_utterances_tokens = []
        if len(previous_utterance_tokens_list) > 0:
            max_previous_utterences_len = max_utterance_len - \
                len(utterance_tokens) - \
                self.prompt_previous_utterances_prefix_length
            if max_previous_utterences_len > 0:
                for previous_utterance_tokens in reversed(previous_utterance_tokens_list):
                    max_previous_utterences_len -= self.utterance_prefix_length + len(previous_utterance_tokens)
                    if max_previous_utterences_len < 0:
                        break
                    previous_utterances_tokens = self.utterance_prefix_tokens + previous_utterance_tokens + previous_utterances_tokens
                if len(previous_utterances_tokens) > 0:
                    previous_utterances_tokens = self.prompt_previous_utterances_prefix_tokens + previous_utterances_tokens

        tokens = self.prompt_prefix_tokens + \
            previous_utterances_tokens + \
            last_utterance_tokens + \
            self.prompt_suffix_tokens
        return tokens
