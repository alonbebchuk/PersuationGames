from transformers import WhisperTokenizer
from typing import Any

PROMPT_PREFIX = """Task:
Determine whether the last utterance falls under the {strategy} strategy.

Strategy Definition:
{strategy} - {strategy_definition}.

"""
PROMPT_PREVIOUS_UTTERANCES = """Previous Utterances:
"""
PROMPT_LAST_UTTERANCE = """Last Utterance:
"""
PROMPT_SUFFIX = """
Strategy: {strategy}

Does the last utterance fall under this strategy?
Answer: {completion}"""
STRATEGY_TO_STRATEGY_DEFINITION = {
    "Identity Declaration": "State one's own role or identity in the game",
    "Accusation": "Claim someone has a specific identity or strategic behavior",
    "Interrogation": "Questions about someone's identity or behavior",
    "Call for Action": "Encourage people to take an action during the game",
    "Defense": "Defending yourself or someone else against an accusation or defending a game-related argument",
    "Evidence": "Provide a body of game-related facts or information",
    "No Strategy": "Any sentences that do not fall into other categories are here. Clarification or discussion of game rules should also be considered 'No-Strategy'"
}


class PromptBuilder:
    def __init__(self, args: Any, strategy: str, tokenizer: WhisperTokenizer):
        self.args = args
        self.strategy = strategy
        self.strategy_definition = STRATEGY_TO_STRATEGY_DEFINITION[strategy]
        self.tokenizer = tokenizer

        prompt_prefix = PROMPT_PREFIX.format(strategy=strategy, strategy_definition=STRATEGY_TO_STRATEGY_DEFINITION[strategy])
        self.prompt_prefix_tokens = ["<|startoftranscript|>", "<|notimestamps|>"] + tokenizer.tokenize(prompt_prefix)
        self.prompt_prefix_length = len(self.prompt_prefix_tokens)

        self.prompt_previous_utterances_tokens = tokenizer.tokenize(PROMPT_PREVIOUS_UTTERANCES)
        self.prompt_previous_utterances_length = len(self.prompt_previous_utterances_tokens)

        self.previous_utterance_prefix_tokens = {i: tokenizer.tokenize(f"{i+1}. ") for i in range(args.context_size)}

        self.prompt_last_utterance_tokens = tokenizer.tokenize(PROMPT_LAST_UTTERANCE)
        self.prompt_last_utterance_length = len(self.prompt_last_utterance_tokens)

        prompt_suffix_no = PROMPT_SUFFIX.format(strategy=strategy, completion="No")
        prompt_suffix_yes = PROMPT_SUFFIX.format(strategy=strategy, completion="Yes")
        self.prompt_suffix_tokens = {
            0: tokenizer.tokenize(prompt_suffix_no) + ["<|endoftext|>"],
            1: tokenizer.tokenize(prompt_suffix_yes) + ["<|endoftext|>"],
        }
        self.prompt_suffix_length = {
            0: len(self.prompt_suffix_no),
            1: len(self.prompt_suffix_yes),
        }

    def build_prompt_tokens(self, label: int, context: list[list[str]], utterance_tokens: list[str]) -> list[str]:
        max_utterance_len = self.args.max_seq_length - \
            self.prompt_prefix_length - \
            self.prompt_last_utterance_length - \
            self.prompt_suffix_length[label]
        utterance_tokens = utterance_tokens[-max_utterance_len:]
        last_utterance_tokens = self.prompt_last_utterance_tokens + utterance_tokens

        max_context_len = max_utterance_len - \
            len(utterance_tokens) - \
            self.prompt_previous_utterances_length
        if max_context_len > 0:
            context_tokens = []
            for i, cxt in enumerate(context[-self.args.context_size:]):
                context_tokens += self.previous_utterance_prefix_tokens[i] + cxt
            context_tokens = context_tokens[-max_context_len:]
            previous_utterances_tokens = self.prompt_previous_utterances_tokens + context_tokens
        else:
            previous_utterances_tokens = []

        tokens = self.prompt_prefix_tokens + \
            previous_utterances_tokens + \
            last_utterance_tokens + \
            self.prompt_suffix_tokens[label]
        return tokens
