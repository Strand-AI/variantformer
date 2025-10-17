"Loss functions for the seq2reg model."

import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def get_loss_fn(loss_fn: str, **kwargs) -> nn.Module:
    "Return the loss function based on the provided string."
    if loss_fn == "cross_entropy":
        return nn.CrossEntropyLoss(reduction=kwargs.get("reduction", "mean"))
    elif loss_fn == "focal":
        return FocalLoss(
            gamma=kwargs.get("gamma", 0), reduction=kwargs.get("reduction", "mean")
        )
    elif loss_fn == "weighted_cross_entropy":
        print("\n\n")
        print("Class weights:")
        print(kwargs.get("class_weight", None))
        return nn.CrossEntropyLoss(
            weight=kwargs.get("class_weight", None),
            reduction=kwargs.get("reduction", "mean"),
        )
    else:
        raise ValueError(f"Invalid loss function: {loss_fn}")
    return None
