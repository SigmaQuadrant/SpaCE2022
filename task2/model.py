from transformers import DebertaPreTrainedModel, DebertaModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class DebertaReader(DebertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
    ):
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)

        # return logits
        # span A start/end B start/end
        # A_start, A_end, B_start, B_end

        A_start_logits, A_end_logits, B_start_logits, B_end_logits = logits[:, :, 0], logits[:, :, 1], logits[:, :, 2], logits[:, :, -1]
        A_start_logits, A_end_logits, B_start_logits, B_end_logits = A_start_logits.contiguous(), A_end_logits.contiguous(), B_start_logits.contiguous(), B_end_logits.contiguous()

        total_loss = None
        if labels is not None:
            A_start_postion, A_end_postion, B_start_postion, B_end_position = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, -1]
            loss = CrossEntropyLoss()

            # assert(A_end_logits.size(1) > torch.max(A_end_postion))

            A_start_loss = loss(A_start_logits, A_start_postion)
            A_end_loss = loss(A_end_logits, A_end_postion)
            B_start_loss = loss(B_start_logits, B_start_postion)
            B_end_loss = loss(B_end_logits, B_end_position)
            total_loss = (A_start_loss + A_end_loss + B_start_loss + B_end_loss) / 4

        return {
            "loss": total_loss,
            "A_start_logits": A_start_logits,
            "A_end_logits": A_end_logits,
            "B_start_logits": B_start_logits,
            "B_end_logits": B_end_logits
        }


if __name__ == '__main__':
    qa_model = DebertaReader.from_pretrained('../pretrained_model/chinese-deberta-large')
    # qa_model.to(torch.device())