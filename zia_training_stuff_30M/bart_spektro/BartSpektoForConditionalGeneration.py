# # coding=utf-8
# # Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """ PyTorch BART model."""
# # import copy
# # import math
# # import random
# # import warnings
# # from typing import Optional, Tuple

# # import torch
# # import torch.utils.checkpoint
# # from torch import nn
# # from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# # from ...activations import ACT2FN
# # from ...file_utils import (
# #     add_code_sample_docstrings,
# #     add_end_docstrings,
# #     add_start_docstrings,
# #     add_start_docstrings_to_model_forward,
# #     replace_return_docstrings,
# # )
# # from ...modeling_outputs import (
# #     BaseModelOutput,
# #     BaseModelOutputWithPastAndCrossAttentions,
# #     CausalLMOutputWithCrossAttentions,
# #     Seq2SeqLMOutput,
# #     Seq2SeqModelOutput,
# #     Seq2SeqQuestionAnsweringModelOutput,
# #     Seq2SeqSequenceClassifierOutput,
# # )
# # from ...modeling_utils import PreTrainedModel
# # from ...utils import logging

# # adam imports
# from ConfigurationBartSpektro import BartSpektroConfig
# from transformers.models.bart.modeling_bart import BartPretrainedModel
# from typing import Optional, Tuple
# import torch
# from torch import nn



# class BartSpektoForConditionalGeneration(BartPretrainedModel):
#     base_model_prefix = "model"
#     _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

#     def __init__(self, config: BartSpektroConfig):
#         super().__init__(config)
#         self.model = BartSpektroModel(config)
#         self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
#         self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_encoder(self):
#         return self.model.get_encoder()

#     def get_decoder(self):
#         return self.model.get_decoder()

#     def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
#         new_embeddings = super().resize_token_embeddings(new_num_tokens)
#         self._resize_final_logits_bias(new_num_tokens)
#         return new_embeddings

#     def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
#         old_num_tokens = self.final_logits_bias.shape[-1]
#         if new_num_tokens <= old_num_tokens:
#             new_bias = self.final_logits_bias[:, :new_num_tokens]
#         else:
#             extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
#             new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
#         self.register_buffer("final_logits_bias", new_bias)

#     def get_output_embeddings(self):
#         return self.lm_head

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         cross_attn_head_mask=None,
#         encoder_outputs=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         decoder_inputs_embeds=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         position_ids=None
#     ):
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#             config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

#         Returns:
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if labels is not None:
#             if use_cache:
#                 logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
#             use_cache = False
#             if decoder_input_ids is None and decoder_inputs_embeds is None:
#                 decoder_input_ids = shift_tokens_right(
#                     labels, self.config.pad_token_id, self.config.decoder_start_token_id
#                 )

#         outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             encoder_outputs=encoder_outputs,
#             decoder_attention_mask=decoder_attention_mask,
#             head_mask=head_mask,
#             decoder_head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             decoder_inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             position_ids=position_ids
#         )
#         lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

#         masked_lm_loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

#         if not return_dict:
#             output = (lm_logits,) + outputs[1:]
#             return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

#         return Seq2SeqLMOutput(
#             loss=masked_lm_loss,
#             logits=lm_logits,
#             past_key_values=outputs.past_key_values,
#             decoder_hidden_states=outputs.decoder_hidden_states,
#             decoder_attentions=outputs.decoder_attentions,
#             cross_attentions=outputs.cross_attentions,
#             encoder_last_hidden_state=outputs.encoder_last_hidden_state,
#             encoder_hidden_states=outputs.encoder_hidden_states,
#             encoder_attentions=outputs.encoder_attentions,
#         )

#     def prepare_inputs_for_generation(
#         self,
#         decoder_input_ids,
#         past=None,
#         attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         cross_attn_head_mask=None,
#         use_cache=None,
#         encoder_outputs=None,
#         **kwargs
#     ):
#         # cut decoder_input_ids if past is used
#         if past is not None:
#             decoder_input_ids = decoder_input_ids[:, -1:]

#         return {
#             "input_ids": None,  # encoder_outputs is defined. input_ids not needed
#             "encoder_outputs": encoder_outputs,
#             "past_key_values": past,
#             "decoder_input_ids": decoder_input_ids,
#             "attention_mask": attention_mask,
#             "head_mask": head_mask,
#             "decoder_head_mask": decoder_head_mask,
#             "cross_attn_head_mask": cross_attn_head_mask,
#             "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
#         }

#     def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
#         return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

#     @staticmethod
#     def _reorder_cache(past, beam_idx):
#         reordered_past = ()
#         for layer_past in past:
#             # cached cross_attention states don't have to be reordered -> they are always the same
#             reordered_past += (
#                 tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
#             )
#         return reordered_past
