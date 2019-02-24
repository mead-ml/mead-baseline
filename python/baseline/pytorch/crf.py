import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline.pytorch.torchy import vec_log_sum_exp, sequence_mask
from baseline.utils import transition_mask as transition_mask_np, Offsets


def transition_mask(vocab, span_type, s_idx, e_idx, pad_idx=None):
    """Create a mask to enforce span sequence transition constraints.

    Returns a Tensor with valid transitions as a 0 and invalid as a 1 for easy use with `masked_fill`
    """
    np_mask = transition_mask_np(vocab, span_type, s_idx, e_idx, pad_idx=pad_idx)
    return torch.from_numpy(np_mask) == 0


class CRF(nn.Module):

    def __init__(self, n_tags, idxs=(Offsets.GO, Offsets.EOS), batch_first=False, constraint=None):
        """Initialize the object.
        :param n_tags: int: The number of tags in your output (emission size)
        :param idxs: Tuple(int. int): The index of the start and stop symbol
            in emissions.
        :param batch_first: bool: if the input [B, T, ...] (true) or [T, B, ...] (false)
        :param constraint: torch.ByteTensor: Constraints on the transitions [N, N]
            invalid transitions should be set to `1`.
        """
        super(CRF, self).__init__()

        self.start_idx, self.end_idx = idxs
        self.n_tags = n_tags
        if constraint is not None:
            self.register_buffer('constraint', constraint.unsqueeze(0))
        else:
            self.constraint = None

        self.transitions_p = nn.Parameter(torch.Tensor(1, self.n_tags, self.n_tags).zero_())
        self.batch_first = batch_first

    def extra_repr(self):
        str_ = "n_tags=%d, batch_first=%s" % (self.n_tags, self.batch_first)
        if self.constraint is not None:
            str_ += ", constrained=True"
        return str_

    @property
    def transitions(self):
        if self.constraint is not None:
            return self.transitions_p.masked_fill(self.constraint, -1e4)
        return self.transitions_p

    def neg_log_loss(self, unary, tags, lengths):
        """Neg Log Loss with a Batched CRF.

        :param unary: torch.FloatTensor: [T, B, N] or [B, T, N]
        :param tags: torch.LongTensor: [T, B] or [B, T]
        :param lengths: torch.LongTensor: [B]

        :return: torch.FloatTensor: [B]
        """
        # Convert from [B, T, N] -> [T, B, N]
        if self.batch_first:
            unary = unary.transpose(0, 1)
            tags = tags.transpose(0, 1)
        _, batch_size, _ = unary.size()
        fwd_score = self.forward(unary, lengths, batch_size)
        gold_score = self.score_sentence(unary, tags, lengths, batch_size)
        return fwd_score - gold_score

    def score_sentence(self, unary, tags, lengths, batch_size):
        """Score a batch of sentences.

        :param unary: torch.FloatTensor: [T, B, N]
        :param tags: torch.LongTensor: [T, B]
        :param lengths: torch.LongTensor: [B]
        :param batch_size: int: B

        :return: torch.FloatTensor: [B]
        """
        trans = self.transitions.squeeze(0)  # [N, N]
        start = torch.full((1, batch_size), self.start_idx, dtype=tags.dtype, device=tags.device)  # [1, B]
        tags = torch.cat([start, tags], 0)  # [T + 1, B]

        # Unfold gives me all slices of size 2 (this tag next tag) from dimension T
        tag_pairs = tags.unfold(0, 2, 1)
        # Move the pair dim to the front and split it into two
        indices = tag_pairs.permute(2, 0, 1).chunk(2)
        trans_score = trans[[indices[1], indices[0]]].squeeze(0)
        # Pull out the values of the tags from the unary scores.
        unary_score = unary.gather(2, tags[1:].unsqueeze(-1)).squeeze(-1)

        mask = sequence_mask(lengths).transpose(0, 1).to(tags.device)
        scores = unary_score + trans_score
        scores = scores.masked_fill(mask == 0, 0)
        scores = scores.sum(0)

        # Add stop tag
        eos_scores = trans[self.end_idx, tags.gather(0, lengths.unsqueeze(0)).squeeze(0)]
        scores = scores + eos_scores
        return scores

    def forward(self, unary, lengths, batch_size):
        """For CRF forward on a batch.

        :param unary: torch.FloatTensor: [T, B, N]
        :param lengths: torch.LongTensor: [B]
        :param batch_size: int: B

        :return: torch.FloatTensor: [B]
        """
        min_length = torch.min(lengths)
        # alphas: [B, 1, N]
        alphas = torch.Tensor(batch_size, 1, self.n_tags).fill_(-1e4).to(unary.device)
        alphas[:, 0, self.start_idx] = 0.
        alphas.requires_grad = True

        trans = self.transitions  # [1, N, N]

        for i, unary_t in enumerate(unary):
            # unary_t: [B, N]
            unary_t = unary_t.unsqueeze(2)  # [B, N, 1]
            # Broadcast alphas along the rows of trans
            # Broadcast trans along the batch of alphas
            # [B, 1, N] + [1, N, N] -> [B, N, N]
            # Broadcast unary_t along the cols of result
            # [B, N, N] + [B, N, 1] -> [B, N, N]
            scores = alphas + trans + unary_t
            new_alphas = vec_log_sum_exp(scores, 2).transpose(1, 2)
            # If we haven't reached your length zero out old alpha and take new one.
            # If we are past your length, zero out new_alpha and keep old one.
            if i >= min_length:
                mask = (i < lengths).view(-1, 1, 1)
                alphas = alphas.masked_fill(mask, 0) + new_alphas.masked_fill(mask == 0, 0)
            else:
                alphas = new_alphas

        terminal_vars = alphas + trans[:, self.end_idx]
        alphas = vec_log_sum_exp(terminal_vars, 2)
        return alphas.view(batch_size)

    def decode(self, unary, lengths):
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N] or [B, T, N]
        :param lengths: torch.LongTensor: [B]

        :return: List[torch.LongTensor]: [B] the paths
        :return: torch.FloatTensor: [B] the path score
        """
        if self.batch_first:
            unary = unary.transpose(0, 1)
        trans = self.transitions  # [1, N, N]
        return viterbi(unary, trans, lengths, self.start_idx, self.end_idx)


def viterbi(unary, trans, lengths, start_idx, end_idx, norm=lambda x, y: x):
    """Do Viterbi decode on a batch.

    :param unary: torch.FloatTensor: [T, B, N]
    :param trans: torch.FloatTensor: [1, N, N]
    :param lengths: torch.LongTensor: [B]
    :param start_idx: int: The index of the go token
    :param end_idx: int: The index of the eos token
    :param norm: Callable: This function should take the initial and a dim to
        normalize along.

    :return: torch.LongTensor: [T, B] the padded paths
    :return: torch.FloatTensor: [B] the path scores
    """
    seq_len, batch_size, tag_size = unary.size()
    min_length = torch.min(lengths)
    backpointers = []

    # Alphas: [B, 1, N]
    alphas = torch.Tensor(batch_size, 1, tag_size).fill_(-1e4).to(unary.device)
    alphas[:, 0, start_idx] = 0
    alphas = norm(alphas, -1)

    for i, unary_t in enumerate(unary):
        next_tag_var = alphas + trans
        viterbi, best_tag_ids = torch.max(next_tag_var, 2)
        backpointers.append(best_tag_ids.data)
        new_alphas = viterbi + unary_t
        new_alphas.unsqueeze_(1)
        if i >= min_length:
            mask = (i < lengths).view(-1, 1, 1)
            alphas = alphas.masked_fill(mask, 0) + new_alphas.masked_fill(mask == 0, 0)
        else:
            alphas = new_alphas

    # Add end tag
    terminal_var = alphas.squeeze(1) + trans[:, end_idx, :]
    path_score, best_tag_id = torch.max(terminal_var, 1)

    # Flip lengths
    rev_len = seq_len - lengths - 1

    best_path = [best_tag_id]
    for i, backpointer_t in enumerate(reversed(backpointers)):
        # Get new best tag candidate
        new_best_tag_id = backpointer_t.gather(1, best_tag_id.unsqueeze(1)).squeeze(1)
        # We are going backwards now, if you haven't passed your flipped length
        # then you aren't in your real results yet so we propagate best tag
        # from the argmax on the terminal_var
        mask = (i > rev_len)
        best_tag_id = best_tag_id.masked_fill(mask, 0) + new_best_tag_id.masked_fill(mask == 0, 0)
        best_path.append(best_tag_id)
    _ = best_path.pop()
    best_path.reverse()
    best_path = torch.stack(best_path)
    # Mask out the extra tags (This might be pointless given that anything that
    # will use this as a dense tensor downstream will mask it itself?)
    seq_mask = sequence_mask(lengths).to(best_path.device).transpose(0, 1)
    best_path = best_path.masked_fill(seq_mask == 0, 0)
    return best_path, path_score
