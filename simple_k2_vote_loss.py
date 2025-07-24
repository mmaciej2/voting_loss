import torch
import k2


def idx_to_seq(idx, length):
    return [int(x) for x in ("{0:0"+str(length)+"b}").format(2**length-1-idx)]


def ind_to_next(i, val, filter_size):
    return (2*(i+1)-1-val)%2**filter_size


def generate_fsa(filter_size, majority_size):
    s_chunk = "0 1 1 0.0\n0 {0} 0 0.0\n"

    seq_ind_to_state = {}
    ind = 1
    for i in range(2 ** filter_size):
        if sum(idx_to_seq(i, filter_size)) >= majority_size:
            seq_ind_to_state[i] = ind
            ind += 1

    for seq_ind in sorted(seq_ind_to_state.keys()):
        for next_symb in [0, 1]:
            next_ind = ind_to_next(seq_ind, next_symb, filter_size)
            if next_ind in seq_ind_to_state:
                s_chunk += f"{seq_ind_to_state[seq_ind]} {seq_ind_to_state[next_ind]} {next_symb} 0.0\n"
        s_chunk += f"{seq_ind_to_state[seq_ind]} {ind} -1 0.0\n"
    s_chunk += f"{ind}"
    s_chunk = s_chunk.format(seq_ind_to_state[1], ind)

    return k2.Fsa.from_str(s_chunk)


class VoteFilterLoss(torch.nn.Module):
    def __init__(
        self,
        filter_size=3,
        majority_size=None,
        filter_dim=-1,
        use_best=False,
        reduction="mean",
    ):
        super().__init__()

        self.filter_size = filter_size
        self.majority_size = majority_size if majority_size is not None else filter_size // 2 + 1
        self.filter_dim = filter_dim
        self.use_best = use_best
        assert reduction in ["mean", "sum"]
        self.reduction = reduction

        self.loss_fn = torch.nn.BCELoss(reduction="none")

    def forward(self, output, target):
        output = output.transpose(self.filter_dim, -1).reshape(-1, output.shape[-1])
        target = target.transpose(self.filter_dim, -1).reshape(-1, target.shape[-1])

        bce_loss = self.loss_fn(output, target)

        # Accumulate loss for annotated negatives
        loss = torch.sum(bce_loss[target == 0])

        # find the onsets/offsets of all positive segments
        padded = torch.nn.functional.pad(target, (1, 1))
        difference = padded[..., 1:] - padded[..., :-1]
        onsets = (difference == 1).nonzero().to(torch.int32)
        offsets = (difference == -1).nonzero().to(torch.int32)
        lengths = offsets - onsets

        sorted_lengths, indices = torch.sort(lengths[:, -1], descending=True)
        sorted_onsets = onsets[indices]
        supervision_segments = torch.cat([sorted_onsets, sorted_lengths.unsqueeze(-1)], dim=-1).cpu()

        log_probs = torch.log(torch.stack([1-output, output], dim=-1))
        dense_fsa_vec = k2.DenseFsaVec(log_probs, supervision_segments)

        median_fsa = generate_fsa(self.filter_size, self.majority_size).to(output.device)
        median_fsa = k2.create_fsa_vec([median_fsa]*len(log_probs))

        lattice = k2.intersect_dense(median_fsa, dense_fsa_vec, output_beam=20)
        loss -= lattice.get_tot_scores(log_semiring=(not self.use_best), use_double_scores=True).sum()

        if self.reduction == "mean":
            loss = loss / torch.numel(output)

        return loss

if __name__ == "__main__":
    target = torch.tensor([[0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 0]]).int()
    output = torch.tensor([[0.1, 0.1, 0.9, 0.4, 0.9, 0.1, 0.1], [0.1, 0.9, 0.9, 0.4, 0.3, 0.9, 0.1]]).float()
    filter_size = 3
    print("filter size:", 3)
    print("labels:\n", target)
    print("predictions:\n", output)
    loss_fn = VoteFilterLoss(filter_size=filter_size, use_best=True)
    loss = loss_fn(output, target.float())
    print("voting loss:\n", loss)
    test_target = torch.tensor([[0, 0, 1, 0, 1, 0, 0], [0, 1, 1, 1, 0, 1, 0]]).float()
    print("oracle permutations:\n", test_target)
    bceloss = torch.nn.BCELoss()
    loss = bceloss(output, test_target.float())
    print("BCE loss:\n", loss)
