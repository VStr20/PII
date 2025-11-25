import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# optional: pip install pytorch-crf
try:
    from torchcrf import CRF
    _HAS_CRF = True
except Exception:
    _HAS_CRF = False


class TokenNERModel(nn.Module):
    """
    Token-level NER model with optional CRF on top.
    - backbone: any HF Transformer encoder (BERT, RoBERTa, DeBERTa, etc.)
    - classifier: linear layer projecting to num_labels
    - crf: optional CRF for better span consistency (recommended for noisy STT)
    """
    def __init__(self, model_name: str, num_labels: int, use_crf: bool = True, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.use_crf = use_crf and _HAS_CRF
        if use_crf and not _HAS_CRF:
            # Don't silently fail — explicitly warn so the caller can `pip install pytorch-crf`.
            raise RuntimeError("CRF requested but torchcrf is not installed. pip install pytorch-crf")

        if self.use_crf:
            self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        """
        Args:
          input_ids, attention_mask, token_type_ids: standard HF inputs (tensors)
          labels: LongTensor of shape (batch, seq_len) with label ids or -100 to ignore (if not using CRF)
        Returns:
          dict with logits (batch, seq_len, num_labels), loss (if labels provided), and optionally `predictions` when using CRF
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

        output = {"logits": logits}

        if labels is not None:
            if self.use_crf:
                # CRF expects mask as bool where True indicates token to include
                mask = attention_mask.type(torch.bool) if attention_mask is not None else torch.ones(labels.shape, dtype=torch.bool, device=labels.device)
                # CRF returns log_likelihood; we want a loss => negative log-likelihood
                # Note: torchcrf's CRF expects emissions shaped (batch, seq_len, num_tags)
                nll = -self.crf(emissions=logits, tags=labels, mask=mask, reduction='mean')
                output["loss"] = nll
            else:
                # Standard token-level cross entropy. Ensure ignored tokens are -100.
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                # reshape to (batch * seq_len, num_labels)
                loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
                output["loss"] = loss

        return output

    def decode(self, logits, attention_mask=None):
        """
        Convert logits to predicted label ids.
        If using CRF, uses CRF.decode(), otherwise argmax.
        Returns a list (batch) of lists (seq_len) of label ids.
        """
        if self.use_crf:
            mask = attention_mask.type(torch.bool) if attention_mask is not None else None
            # CRF.decode returns List[List[int]]
            return self.crf.decode(logits, mask=mask)
        else:
            preds = torch.argmax(logits, dim=-1)
            return preds.tolist()


def create_model(model_name: str, num_labels: int, use_crf: bool = True, dropout: float = 0.1):
    """
    Convenience: create and return TokenNERModel.
    """
    return TokenNERModel(model_name=model_name, num_labels=num_labels, use_crf=use_crf, dropout=dropout)
