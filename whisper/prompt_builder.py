from transformers import WhisperTokenizer
from typing import Any, List


STRATEGY_TO_STRATEGY_DEFINITION = {
    "Identity Declaration": "State one's own role or identity in the game",
    "Accusation": "Claim someone has a specific identity or strategic behavior",
    "Interrogation": "Questions about someone's identity or behavior",
    "Call for Action": "Encourage people to take an action during the game",
    "Defense": "Defending yourself or someone else against an accusation or defending a game-related argument",
    "Evidence": "Provide a body of game-related facts or information",
}


PROMPT_PREFIX_TEMPLATE = """Task: Determine whether the last utterance falls under the {strategy} strategy.
Strategy Definition: {strategy} - {strategy_definition}.
"""
PROMPT_UTTERANCE_PREFIX = "Utterance -> "
PROMPT_PREVIOUS_UTTERANCES_PREFIX = """Previous Utterances:
"""
PROMPT_LAST_UTTERANCE_PREFIX = f"""Last Utterance:
{PROMPT_UTTERANCE_PREFIX}"""
PROMPT_SUFFIX_TEMPLATE = """
Strategy: {strategy}
Does the last utterance fall under this strategy?
Answer: """


class PromptBuilder:
    def __init__(self, args: Any, tokenizer: WhisperTokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.prompt_prefix_tokens_dict = {"No Strategy": ["<|startoftranscript|>", "<|notimestamps|>"]}
        self.prompt_suffix_tokens_dict = {"No Strategy": []}
        for strategy, strategy_definition in STRATEGY_TO_STRATEGY_DEFINITION.items():
            prompt_prefix = PROMPT_PREFIX_TEMPLATE.format(strategy=strategy, strategy_definition=strategy_definition)
            prompt_suffix = PROMPT_SUFFIX_TEMPLATE.format(strategy=strategy)
            self.prompt_prefix_tokens_dict[strategy] = ["<|startoftranscript|>", "<|notimestamps|>"] + tokenizer.tokenize(prompt_prefix)
            self.prompt_suffix_tokens_dict[strategy] = tokenizer.tokenize(prompt_suffix)

        self.utterance_prefix_tokens = tokenizer.tokenize(PROMPT_UTTERANCE_PREFIX)
        self.utterance_prefix_length = len(self.utterance_prefix_tokens)

        self.prompt_previous_utterances_prefix_tokens = tokenizer.tokenize(PROMPT_PREVIOUS_UTTERANCES_PREFIX)
        self.prompt_previous_utterances_prefix_length = len(self.prompt_previous_utterances_prefix_tokens)

        self.prompt_last_utterance_prefix_tokens = tokenizer.tokenize(PROMPT_LAST_UTTERANCE_PREFIX)
        self.prompt_last_utterance_prefix_length = len(self.prompt_last_utterance_prefix_tokens)

    def get_strategy_prompt_tokens(self, strategy: str, previous_utterance_tokens_list: List[List[str]], utterance_tokens: List[str]) -> List[str]:
        prompt_prefix_tokens = self.prompt_prefix_tokens_dict[strategy]
        prompt_suffix_tokens = self.prompt_suffix_tokens_dict[strategy]

        prompt_prefix_length = len(prompt_prefix_tokens)
        prompt_suffix_length = len(prompt_suffix_tokens)

        max_utterance_len = self.args.max_seq_length - prompt_prefix_length - prompt_suffix_length - self.prompt_last_utterance_prefix_length
        utterance_tokens = utterance_tokens[-max_utterance_len:]
        last_utterance_tokens = self.prompt_last_utterance_prefix_tokens + utterance_tokens

        previous_utterances_tokens = []
        if len(previous_utterance_tokens_list) > 0:
            used_length = len(last_utterance_tokens)
            max_previous_utterences_len = self.args.max_seq_length - prompt_prefix_length - prompt_suffix_length - used_length - self.prompt_previous_utterances_prefix_length

            if max_previous_utterences_len > 0:
                for previous_utterance_tokens in reversed(previous_utterance_tokens_list):
                    utterance_with_prefix_length = self.utterance_prefix_length + len(previous_utterance_tokens)
                    if max_previous_utterences_len - utterance_with_prefix_length < 0:
                        break
                    previous_utterances_tokens = self.utterance_prefix_tokens + previous_utterance_tokens + previous_utterances_tokens
                    max_previous_utterences_len -= utterance_with_prefix_length

                if len(previous_utterances_tokens) > 0:
                    previous_utterances_tokens = self.prompt_previous_utterances_prefix_tokens + previous_utterances_tokens

        tokens = prompt_prefix_tokens + previous_utterances_tokens + last_utterance_tokens + prompt_suffix_tokens
        return tokens

    def get_prompt_tokens(self, previous_utterance_tokens_list: List[List[str]], utterance_tokens: List[str]) -> List[str]:
        return self.get_strategy_prompt_tokens("No Strategy", previous_utterance_tokens_list, utterance_tokens)
