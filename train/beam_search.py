"""Beam Search decoder for sequence generation."""

import torch
from dataclasses import dataclass


@dataclass
class BeamHypothesis:
    tokens: list
    score: float


class BeamSearch:
    def __init__(self, beam_size: int = 5, max_len: int = 128,
                 length_penalty: float = 1.0,
                 sos_idx: int = 1, eos_idx: int = 2, pad_idx: int = 0):
        self.beam_size = beam_size
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

    @torch.no_grad()
    def search(self, model, src_input_ids: torch.Tensor,
               src_attention_mask: torch.Tensor):
        """
        Args:
            model: the Seq2Seq model (must implement .encode() and a
                   step-wise .decode_step(memory, memory_mask, tgt_ids))
            src_input_ids: (1, L_src) — single sentence
            src_attention_mask: (1, L_src)
        Returns:
            list of BeamHypothesis sorted by score descending
        """
        model.eval()
        device = src_input_ids.device
        memory, memory_mask = model.encode(src_input_ids, src_attention_mask)
        # memory: (1, L_src, d_model), memory_mask: (1, L_src)

        # Each beam: (tokens, log_prob)
        beams = [([self.sos_idx], 0.0)]
        done_beams = []

        for step in range(self.max_len):
            candidates = []

            for tokens, log_prob in beams:
                if tokens[-1] == self.eos_idx:
                    # Already finished — keep as-is for comparison, but don't
                    # extend further. We DON'T add to candidates; keep in
                    # done (will be re-added below if still top beam_size).
                    done_beams.append(BeamHypothesis(tokens, log_prob))
                    continue

                tgt_ids = torch.tensor([tokens], device=device)  # (1, L_dec)
                logits = model.decode_step(memory, memory_mask, tgt_ids)
                # logits: (1, 1, vocab_size)
                next_log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)

                # Top-k over vocabulary
                topk_scores, topk_ids = torch.topk(next_log_probs, self.beam_size)

                for score, token_id in zip(topk_scores.tolist(), topk_ids.tolist()):
                    new_tokens = tokens + [token_id]
                    new_score = log_prob + score
                    candidates.append((new_tokens, new_score))

            # Prune: keep top beam_size candidates by normalized score
            candidates.sort(key=lambda x: self._norm_score(x[0], x[1]), reverse=True)

            # Also consider done beams when pruning
            all_candidates = candidates + [
                (h.tokens, h.score) for h in done_beams
            ]
            all_candidates.sort(key=lambda x: self._norm_score(x[0], x[1]),
                                reverse=True)
            beams = all_candidates[:self.beam_size]

            # Check if all beams finished
            if all(t[-1] == self.eos_idx for t, _ in beams):
                break

        # Collect results
        results = [BeamHypothesis(t, s) for t, s in beams]
        # Re-sort final results
        results.sort(key=lambda h: self._norm_score(h.tokens, h.score),
                     reverse=True)
        return results

    def _norm_score(self, tokens: list, score: float) -> float:
        """Length-normalised log-probability."""
        length = len(tokens)  # includes SOS
        return score / (length ** self.length_penalty)
