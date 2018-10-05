import torch
from baseline.utils import crf_mask as crf_m
import torch.autograd
import torch.nn as nn
from baseline.pytorch.torchy import vec_log_sum_exp


def crf_mask(vocab, span_type, s_idx, e_idx, pad_idx=None):
    """Create a CRF mask.

    Returns a Tensor with valid transitions as a 0 and invalid as a 1 for easy use with `masked_fill`
    """
    np_mask = crf_m(vocab, span_type, s_idx, e_idx, pad_idx=pad_idx)
    return torch.from_numpy(np_mask) == 0

class CRF(nn.Module):
    def __init__(self, n_tags, idxs=None, batch_first=True, vocab=None, span_type=None, pad_idx=None):
        """Initialize the object.
        :param n_tags: int The number of tags in your output (emission size)
        :param idxs: Tuple(int. int) The index of the start and stop symbol
            in emissions.
        :param vocab: The label vocab of the form vocab[string]: int
        :param span_type: The tagging span_type used. `IOB`, `IOB2`, or `IOBES`
        :param pds_idx: The index of the pad symbol in the vocab
        Note:
            if idxs is none then the CRF adds these symbols to the emission
            vectors and n_tags is assumed to be the number of output tags.
            if idxs is not none then the first element is assumed to be the
            start index and the second idx is assumed to be the end index. In
            this case n_tags is assumed to include the start and end symbols.
            if vocab is not None then a transition mask will be created that
            limits illegal transitions.
        """
        super(CRF, self).__init__()

        if idxs is None:
            self.start_idx = n_tags
            self.end_idx = n_tags + 1
            self.n_tags = n_tags + 2
            self.add_ends = True
        else:
            self.start_idx, self.end_idx = idxs
            self.n_tags = n_tags
            self.add_ends = False
        self.span_type = None
        if vocab is not None:
            assert span_type is not None, "To mask transitions you need to provide a tagging span_type, choices are `IOB`, `BIO` (or `IOB2`), and `IOBES`"
            # If there weren't start and end idx provided we need to add them.
            if idxs is None:
                vocab = vocab.copy()
                vocab['<GO>'] = self.start_idx
                vocab['<EOS>'] = self.end_idx
            self.span_type = span_type
            self.register_buffer('mask', crf_mask(vocab, span_type, self.start_idx, self.end_idx, pad_idx).unsqueeze(0))
        else:
            self.mask = None

        self.transitions_p = nn.Parameter(torch.Tensor(1, self.n_tags, self.n_tags).zero_())
        self.batch_first = batch_first

    def extra_repr(self):
        str_ = "n_tags=%d, batch_first=%s" % (self.n_tags, self.batch_first)
        if self.mask is not None:
            str_ += ", masked=True, span_type=%s" % self.span_type
        return str_

    @staticmethod
    def _prep_input(input_):
        ends = torch.Tensor(input_.size()[0], input_.size()[1], 2).fill_(-1e4).to(input_.device)
        return torch.cat([input_, ends], dim=2)

    @property
    def transitions(self):
        if self.mask is not None:
            return self.transitions_p.masked_fill(self.mask, -1e4)
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
        if self.add_ends:
            unary = CRF._prep_input(unary)
        _, batch_size, _ = unary.size()
        min_lengths = torch.min(lengths)
        fwd_score = self.forward(unary, lengths, batch_size, min_lengths)
        gold_score = self.score_sentence(unary, tags, lengths, batch_size, min_lengths)
        return fwd_score - gold_score

    def score_sentence(self, unary, tags, lengths, batch_size, min_length):
        """Score a batch of sentences.

        :param unary: torch.FloatTensor: [T, B, N]
        :param tags: torch.LongTensor: [T, B]
        :param lengths: torch.LongTensor: [B]
        :param batzh_size: int: B
        :param min_length: torch.LongTensor: []

        :return: torch.FloatTensor: [B]
        """
        trans = self.transitions.squeeze(0)  # [N, N]
        batch_range = torch.arange(batch_size, dtype=torch.int64)  # [B]
        start = torch.full((1, batch_size), self.start_idx, dtype=tags.dtype, device=tags.device)  # [1, B]
        tags = torch.cat([start, tags], 0)  # [T, B]
        scores = torch.zeros(batch_size, requires_grad=True).to(unary.device)  # [B]
        for i, unary_t in enumerate(unary):
            new_scores = (
                trans[tags[i + 1], tags[i]] +
                unary_t[batch_range, tags[i + 1]]
            )
            if i >= min_length:
                # If we are farther along `T` than your length don't add to your score
                mask = (i >= lengths)
                scores = scores + new_scores.masked_fill(mask, 0)
            else:
                scores = scores + new_scores
        # Add stop tag
        scores = scores + trans[self.end_idx, tags[lengths, batch_range]]
        return scores

    def forward(self, unary, lengths, batch_size, min_length):
        """For CRF forward on a batch.

        :param unary: torch.FloatTensor: [T, B, N]
        :param lengths: torch.LongTensor: [B]
        :param batzh_size: int: B
        :param min_length: torch.LongTensor: []

        :return: torch.FloatTensor: [B]
        """
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
        return alphas.squeeze()

    def decode(self, unary, lengths):
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N] or [B, T, N]
        :param lengths: torch.LongTensor: [B]

        :return: List[torch.LongTensor]: [B] the paths
        :return: torch.FloatTensor: [B] the path score
        """
        if self.batch_first:
            unary = unary.transpose(0, 1)
        if self.add_ends:
            unary = CRF._prep_input(unary)
        seq_len, batch_size, _ = unary.size()
        min_length = torch.min(lengths)
        batch_range = torch.arange(batch_size, dtype=torch.int64)
        backpointers = []

        # alphas: [B, 1, N]
        alphas = torch.Tensor(batch_size, 1, self.n_tags).fill_(-1e4).to(unary.device)
        alphas[:, 0, self.start_idx] = 0
        alphas.requires_grad = True

        trans = self.transitions  # [1, N, N]

        for i, unary_t in enumerate(unary):
            # Broadcast alphas along the rows of trans and trans along the batch of alphas
            next_tag_var = alphas + trans  # [B, 1, N] + [1, N, N] -> [B, N, N]
            viterbi, best_tag_ids = torch.max(next_tag_var, 2) # [B, N]
            backpointers.append(best_tag_ids.data)
            new_alphas = viterbi + unary_t  # [B, N] + [B, N]
            new_alphas.unsqueeze_(1)  # Prep for next round
            # If we haven't reached your length zero out old alpha and take new one.
            # If we are past your length, zero out new_alpha and keep old one.
            if i >= min_length:
                mask = (i < lengths).view(-1, 1, 1)
                alphas = alphas.masked_fill(mask, 0) + new_alphas.masked_fill(mask == 0, 0)
            else:
                alphas = new_alphas

        # Add end tag
        terminal_var = alphas.squeeze(1) + trans[:, self.end_idx]
        _, best_tag_id = torch.max(terminal_var, 1)
        path_score = terminal_var[batch_range, best_tag_id]  # Select best_tag from each batch

        best_path = [best_tag_id]
        # Flip lengths
        rev_len = seq_len - lengths - 1
        for i, backpointer_t in enumerate(reversed(backpointers)):
            # Get new best tag candidate
            new_best_tag_id = backpointer_t[batch_range, best_tag_id]
            # We are going backwards now, if you passed your flipped length then you aren't in your real results yet
            mask = (i > rev_len)
            best_tag_id = best_tag_id.masked_fill(mask, 0) + new_best_tag_id.masked_fill(mask == 0, 0)
            best_path.append(best_tag_id)
        _ = best_path.pop()
        best_path.reverse()
        best_path = torch.stack(best_path)
        # Return list of paths
        paths = []
        best_path = best_path.transpose(0, 1)
        for path, length in zip(best_path, lengths):
            paths.append(path[:length])
        return paths, path_score.squeeze(0)
