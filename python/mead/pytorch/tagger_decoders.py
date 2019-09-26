import torch
import torch.nn as nn


class InferenceCRF(torch.jit.ScriptModule):
    __constants__ = ['start_idx', 'end_idx', 'batch_first']

    def __init__(self, transitions, start_idx, end_idx, batch_first=False):
        super(InferenceCRF, self).__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.transitions = torch.nn.Parameter(transitions)
        self.batch_first = batch_first

    @torch.jit.script_method
    def decode(self, unary, length):
        if self.batch_first:
            unary = unary.transpose(0, 1)
        unary = unary.squeeze(1)
        return script_viterbi(unary, self.transitions, self.start_idx, self.end_idx)

    # @torch.jit.script_method
    # def viterbi(self, unary):
    #     return script_viterbi(unary, self.transitions, self.start_idx, self.end_idx)


class InferenceGreedyDecoder(nn.Module):
    def decode(self, unary, length):
        _, path = torch.max(unary, dim=2)
        return path, _


@torch.jit.script
def script_viterbi(unary, trans, start_idx, end_idx):
    # type: (Tensor, Tensor, int, int) -> Tuple[Tensor, Tensor]
    backpointers = []
    alphas = torch.full((1, unary.size(1)), -1e4, dtype=unary.dtype, device=unary.device)
    alphas[0, start_idx] = 0

    for i in range(unary.size(0)):
        unary_t = unary[i, :]
        next_tag_var = alphas + trans
        viterbi, best_tag_ids = torch.max(next_tag_var, 1)
        backpointers.append(best_tag_ids)
        alphas = viterbi + unary_t
        alphas = alphas.unsqueeze(0)

    terminal_vars = alphas.squeeze(0) + trans[end_idx, :]
    path_score, best_tag_id = torch.max(terminal_vars, 0)

    best_path = [best_tag_id]
    for i in range(len(backpointers)):
        i = len(backpointers) - i - 1
        best_tag_id = backpointers[i][best_tag_id]
        best_path.append(best_tag_id)

    new_path = []
    for i in range(len(best_path)):
        i = len(best_path) - i - 1
        new_path.append(best_path[i])
    return torch.stack(new_path[1:]), path_score


if __name__ == '__main__':
    from eight_mile.pytorch.layers import CRF, transition_mask
    vocab = ["<GO>", "<EOS>", "B-X", "I-X", "E-X", "S-X", "O", "B-Y", "I-Y", "E-Y", "S-Y"]
    vocab = {k: i for i, k in enumerate(vocab)}
    mask = transition_mask(vocab, "IOBES", 0, 1)
    crf = CRF(10, (0, 1), batch_first=False)
    trans = crf.transitions

    icrf = InferenceCRF(torch.nn.Parameter(trans.squeeze(0)), 0, 1, False)

    u = torch.rand(20, 1, 10)
    l = torch.full((1,), 20, dtype=torch.long)
    print(crf.decode(u, l))
    print(icrf.decode(u, l))

    u = torch.rand(15, 1, 10)
    traced_model = torch.jit.trace(icrf.decode, (u, l))
    traced_model.save('crf.pt')
    traced_model = torch.jit.load('crf.pt')

    u = torch.rand(8, 1, 10)
    l = torch.full((1,), 8, dtype=torch.long)

    print(crf.decode(u, l))
    print(traced_model(u, l))

    u = torch.rand(22, 1, 10)
    l = torch.full((1,), 22, dtype=torch.long)

    print(crf.decode(u, l))
    print(traced_model(u, l))
