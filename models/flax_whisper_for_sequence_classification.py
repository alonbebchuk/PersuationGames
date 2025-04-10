import jax.numpy as jnp
import flax.linen as nn
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_flax_whisper import FlaxWhisperModule, FlaxWhisperPreTrainedModel, FlaxSequenceClassifierOutput


class FlaxWhisperForSequenceClassificationModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.whisper = FlaxWhisperModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_features,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        decoder_position_ids=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # Model
        outputs = self.whisper(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # Get the last hidden state from the encoder
        encoder_last_hidden_state = outputs.encoder_last_hidden_state if return_dict else outputs[0]
        
        # Use the mean of the last hidden state as the pooled output
        pooled_output = jnp.mean(encoder_last_hidden_state, axis=1)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        
        # Apply classifier
        logits = self.classifier(pooled_output)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )


class FlaxWhisperForSequenceClassification(FlaxWhisperPreTrainedModel):
    module_class = FlaxWhisperForSequenceClassificationModule
