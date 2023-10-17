from transformers.modeling_roberta import *


class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = torch.nn.GELU()
        self.up = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.LayerNorm(x + self.dropout(self.up(self.act(self.down(x)))))


class RobertaForQuestionAnswering(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        config.output_hidden_states = True
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # Add adapters.
        self.adapter = nn.ModuleList([Adapter(config) for _ in range(config.num_hidden_layers // 2)])

        # Add side block output head.
        self.adapter_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing.
        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        layer_outputs = outputs[2]

        # Compute the output of the backbone.
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Initialize the side block state as the output of the embedding.
        adapter_states = layer_outputs[0]
        # Side block forward propagation.
        for i in range(1, len(layer_outputs), 2):
            adapter_states = self.adapter[i // 2](adapter_states + layer_outputs[i] + layer_outputs[i + 1])

        # Compute the output of the side block.
        adapter_logits = self.adapter_outputs(adapter_states)
        adapter_start_logits, adapter_end_logits = adapter_logits.split(1, dim=-1)
        adapter_start_logits = adapter_start_logits.squeeze(-1).contiguous()
        adapter_end_logits = adapter_end_logits.squeeze(-1).contiguous()

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension.
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms.
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # Compute loss for warmup, we only consider the output of the side block.
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(adapter_start_logits, start_positions)
            end_loss = loss_fct(adapter_end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        outputs = (start_logits, end_logits), (adapter_start_logits, adapter_end_logits)
        return outputs
