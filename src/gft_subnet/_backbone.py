"""Last hidden state plus logits on demand, with bounded vocabulary activations."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class Backbone:
    """Run the decoder once and apply the LM head only where logits are needed.

    Uses ``model.get_decoder()`` and ``model.get_output_embeddings()`` when a
    causal LM exposes them, so no per-layer hidden states are kept. Falls back
    to the full model with ``output_hidden_states=True``.
    """

    def __init__(self, model):
        self.model = model
        self.decoder = None
        self.head = None
        try:
            decoder = model.get_decoder() if hasattr(model, "get_decoder") else None
            head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        except (AttributeError, NotImplementedError, TypeError):
            decoder = head = None
        if decoder is not None and head is not None and decoder is not model:
            self.decoder, self.head = decoder, head

    def run(self, **kwargs):
        """Return ``(last_hidden [B, T, H], past_key_values, full_logits)``."""
        if self.decoder is not None:
            outputs = self.decoder(**kwargs)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden = outputs[0]
            return hidden, getattr(outputs, "past_key_values", None), None
        outputs = self.model(output_hidden_states=True, **kwargs)
        return outputs.hidden_states[-1], getattr(outputs, "past_key_values", None), outputs.logits

    def logits_at(self, hidden: torch.Tensor, full_logits, positions) -> torch.Tensor:
        """Logits at ``positions`` (int or 1-D index tensor) of every row."""
        if self.head is not None:
            return self.head(hidden[:, positions]).float()
        return full_logits[:, positions].float()

    def token_logps(self, hidden, full_logits, positions, targets, *,
                    temperature=1.0, chunk_size=128, return_entropy=False):
        """Score selected tokens without retaining a sequence x vocab softmax.

        The LM head is checkpointed together with cross entropy, so backward
        also materializes at most one chunk of vocabulary logits.
        """
        features = (hidden if self.head is not None else full_logits)[:, positions]

        def score(block, ids):
            logits = self.head(block) if self.head is not None else block
            logits = logits.float()
            if temperature > 1e-8:
                logits = logits / temperature
            logps = -F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), ids.reshape(-1),
                reduction="none").reshape(ids.shape)
            if not return_entropy:
                return logps
            with torch.no_grad():
                distribution = logits.log_softmax(-1)
                entropy = -(distribution.exp() * distribution).sum(-1)
            return logps, entropy

        chunks = []
        for start in range(0, targets.shape[1], chunk_size):
            block = features[:, start:start + chunk_size]
            ids = targets[:, start:start + chunk_size]
            chunks.append(checkpoint(score, block, ids, use_reentrant=False)
                          if torch.is_grad_enabled() else score(block, ids))
        if return_entropy:
            return tuple(torch.cat(parts, dim=1) for parts in zip(*chunks, strict=True))
        return torch.cat(chunks, dim=1)
