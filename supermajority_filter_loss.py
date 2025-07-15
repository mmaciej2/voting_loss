import torch
from sortedcontainers import SortedDict
from itertools import combinations


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


class MedianFilterLoss(torch.nn.Module):
    def __init__(
        self,
        loss,
        filter_size=3,
        majority_size=None,
        reduction="mean",
        return_target=False,
    ):
        super().__init__()

        self.loss_fn = loss
        self.filter_size = filter_size
        self.majority_size = majority_size if majority_size is not None else filter_size // 2 + 1
        assert reduction in ["mean", "sum"]
        self.reduction = reduction
        self.return_target = return_target

        self.loss_fn.reduction = "none"

    def forward(self, output, target):
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
                new_paths = SortedDict()
                if sub_one_loss[i] < sub_zero_loss[i]:
                    for inter_loss, inter_path in paths.items():
                        new_paths[inter_loss + sub_one_loss[i]] = inter_path + [1]
                        if self.majority_size == 1 or sum(inter_path[-self.majority_size+1:]) == self.majority_size-1:
                            break  # this path has lowest loss and can never be pruned, so we are done
                else: # sub_zero_loss[i] < sub_one_loss[i]
                    new_max_loss = torch.inf
                    for inter_loss, inter_path in paths.items():
                        if sum(inter_path[-filter_size+1:]) >= self.majority_size:
                            if inter_loss + sub_zero_loss[i] > new_max_loss:
                                break # worse than an unprunable path, so we are done
                            new_paths[inter_loss + sub_zero_loss[i]] = inter_path + [0]
                        if inter_loss + sub_one_loss[i] < new_max_loss:
                            new_paths[inter_loss + sub_one_loss[i]] = inter_path + [1]
                            if self.majority_size == 1 or sum(inter_path[-self.majority_size+1:]) == self.majority_size-1:
                                new_max_loss = inter_loss + sub_one_loss[i]
                paths = new_paths

            seg_loss, path = paths.peekitem(0)
            loss += seg_loss
            if self.return_target:
                new_target[*batch_etc, onset:offset] = torch.tensor(path)

        if self.reduction == "mean":
            loss = loss / torch.numel(output)

        if self.return_target:
            return loss, new_target
        else:
            return loss


if __name__ == "__main__":
    loss_fn = MedianFilterLoss(torch.nn.BCELoss(reduction="none"),
                               filter_size=3,
                               majority_size=2,
                               return_target=True)

    target = torch.tensor([[0, 0, 1, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1, 0]]).int()
    output = torch.tensor([[0.1, 0.1, 0.9, 0.4, 0.9, 0.1, 0.9], [0.1, 0.9, 0.9, 0.4, 0.3, 0.9, 0.1]]).float()
    loss, new_target = loss_fn(output, target)
    print(target)
    print(output)
    print(new_target)
    print("loss:", loss)
    test_fn = torch.nn.BCELoss()
    print("loss:", test_fn(output, new_target.float()))

#    target = torch.zeros(2, 1000)
#    target[:, 200:300] = 1
#    target[:, 500:550] = 1
#    target[:, 700:800] = 1
#
#    output = torch.clone(target)
#    factor = 1.0
#    output[:, 0:200] += torch.rand(2, 200)/factor
#    output[:, 200:300] -= torch.rand(2, 100)/factor
#    output[:, 300:500] += torch.rand(2, 200)/factor
#    output[:, 500:550] -= torch.rand(2, 50)/factor
#    output[:, 550:700] += torch.rand(2, 150)/factor
#    output[:, 700:800] -= torch.rand(2, 100)/factor
#    output[:, 800:1000] += torch.rand(2, 200)/factor
#    loss, new_target = loss_fn(output, target)
#    print(target[:, 200:210])
#    print(output[:, 200:210])
#    print(new_target[:, 200:210])
#    print("loss:", loss)
#    test_fn = torch.nn.BCELoss()
#    print("loss:", test_fn(output, new_target.float()))
