import torch
from sortedcontainers import SortedDict
from itertools import combinations
from functools import partial


def initialize_paths(paths, one_loss, zero_loss, majority_size):
    length = len(one_loss)

    loss_diff = zero_loss - one_loss

    # compute best unprunable seqeunce (second half all ones)
    max_loss, max_seq = torch.min(torch.stack([zero_loss, one_loss])[:, :-majority_size], dim=0)
    max_loss = torch.sum(max_loss) + torch.sum(one_loss[-majority_size:])
    max_seq = max_seq.tolist() + [1]*majority_size
    paths[max_loss] = max_seq

    # check all target bit flips that don't violate median filter and are better than best unprunable
    inds = (loss_diff < 0).nonzero()[:, 0].tolist()
    for n in range(length-majority_size):
        for comb in combinations(inds, n+1):
            comb_loss = 0
            comb_seq = [1] * length
            for i in range(length):
                if i in comb:
                    comb_loss += zero_loss[i]
                    comb_seq[i] = 0
                else:
                    comb_loss += one_loss[i]
            if comb_loss < max_loss:
                paths[comb_loss] = comb_seq


class SupermajorityFilterLoss(torch.nn.Module):
    def __init__(
        self,
        loss,
        filter_size=3,
        majority_size=None,
        filter_dim=-1,
        reduction="mean",
        return_target=False,
    ):
        super().__init__()

        self.loss_fn = loss
        self.filter_size = filter_size
        self.majority_size = majority_size if majority_size is not None else filter_size // 2 + 1
        self.filter_dim = filter_dim
        assert reduction in ["mean", "sum"]
        self.reduction = reduction
        self.return_target = return_target

        self.loss_fn.reduction = "none"

    def forward(self, output, target):
        output = output.transpose(self.filter_dim, -1)
        target = target.transpose(self.filter_dim, -1)
        if self.return_target:
            new_target = target.clone()

        zero_loss = self.loss_fn(output, torch.zeros_like(output))
        one_loss = self.loss_fn(output, torch.ones_like(output))

        # Accumulate loss for annotated negatives
        loss = torch.sum(zero_loss[target == 0])

        # find the onsets/offsets of all positive segments
        padded = torch.nn.functional.pad(target, (1, 1))
        difference = padded[..., 1:] - padded[..., :-1]
        onsets = (difference == 1).nonzero()
        offsets = (difference == -1).nonzero()

        for onset, offset in zip(onsets, offsets):
            batch_etc = onset[:-1]
            onset = onset[-1]
            offset = offset[-1]
            # only look at loss in current segment
            sub_one_loss = one_loss[*batch_etc, onset:offset]
            sub_zero_loss = zero_loss[*batch_etc, onset:offset]

            length = offset - onset
            if length < self.majority_size:  # short segments are untouched
                loss += torch.sum(sub_one_loss)
                continue

            if length < self.filter_size:
                filter_size = length
            else:
                filter_size = self.filter_size

            paths = SortedDict()
            initialize_paths(paths, sub_one_loss[:filter_size], sub_zero_loss[:filter_size], self.majority_size)

            for i in range(filter_size, length):
                suffix_to_paths = {} # We keep track of all new paths that end in the same pattern and only keep best
                if sub_one_loss[i] < sub_zero_loss[i]:
                    for inter_loss, inter_path in paths.items():
                        new_loss = inter_loss + sub_one_loss[i]
                        new_path = inter_path + [1]
                        try:
                            suffix_to_paths[str(new_path[-filter_size:])][new_loss] = new_path
                        except KeyError:
                            suffix_to_paths[str(new_path[-filter_size:])] = SortedDict()
                            suffix_to_paths[str(new_path[-filter_size:])][new_loss] = new_path
                        if self.majority_size == 1 or sum(inter_path[-self.majority_size+1:]) == self.majority_size-1:
                            break  # this path has lowest loss and can never be pruned, so we are done

                else: # sub_zero_loss[i] < sub_one_loss[i]
                    new_max_loss = torch.inf
                    for inter_loss, inter_path in paths.items():
                        if sum(inter_path[-filter_size+1:]) >= self.majority_size:
                            if inter_loss + sub_zero_loss[i] > new_max_loss:
                                break # worse than an unprunable path, so we are done
                            new_loss = inter_loss + sub_zero_loss[i]
                            new_path = inter_path + [0]
                            try:
                                suffix_to_paths[str(new_path[-filter_size:])][new_loss] = new_path
                            except KeyError:
                                suffix_to_paths[str(new_path[-filter_size:])] = SortedDict()
                                suffix_to_paths[str(new_path[-filter_size:])][new_loss] = new_path
                        if inter_loss + sub_one_loss[i] < new_max_loss:
                            new_loss = inter_loss + sub_one_loss[i]
                            new_path = inter_path + [1]
                            try:
                                suffix_to_paths[str(new_path[-filter_size:])][new_loss] = new_path
                            except KeyError:
                                suffix_to_paths[str(new_path[-filter_size:])] = SortedDict()
                                suffix_to_paths[str(new_path[-filter_size:])][new_loss] = new_path
                            if self.majority_size == 1 or sum(inter_path[-self.majority_size+1:]) == self.majority_size-1:
                                new_max_loss = inter_loss + sub_one_loss[i]

                paths = SortedDict()
                for suffix_paths in suffix_to_paths.values():
                    best_path = suffix_paths.peekitem(0)
                    paths[best_path[0]] = best_path[1]

            seg_loss, path = paths.peekitem(0)
            loss += seg_loss
            if self.return_target:
                new_target[*batch_etc, onset:offset] = torch.tensor(path)

        if self.reduction == "mean":
            loss = loss / torch.numel(output)

        if self.return_target:
            return loss, new_target.transpose(self.filter_dim, -1)
        else:
            return loss


MedianFilterLoss = partial(SupermajorityFilterLoss, majority_size=None)
