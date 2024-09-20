import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def sample_ellipse_fast(x, y, r1, r2, count=32, dt=0.01):
    batch_size, num_el = r1.shape
    device = r1.device
    num_integrals = int(round(2 * math.pi / dt))

    thetas = dt * torch.arange(num_integrals, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_el, 1)
    thetas_c = torch.cumsum(thetas, dim=-1)
    dpt = torch.sqrt((r1.unsqueeze(-1) * torch.sin(thetas_c)) ** 2 + (r2.unsqueeze(-1) * torch.cos(thetas_c)) ** 2)
    circumference = dpt.sum(dim=-1)

    run = torch.cumsum(
        torch.sqrt((r1.unsqueeze(-1) * torch.sin(thetas + dt)) ** 2 + (r2.unsqueeze(-1) * torch.cos(thetas + dt)) ** 2),
        dim=-1)
    sub = (count * run) / circumference.unsqueeze(-1)

    # OK, now find the smallest point >= 0..count-1
    counts = torch.arange(count, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_el,
                                                                                              num_integrals, 1)
    diff = sub.unsqueeze(dim=-1) - counts
    diff[diff < 0] = 10000.0

    idx = diff.argmin(dim=2)
    thetas = torch.gather(thetas + dt, -1, idx)

    xy = torch.stack((x.unsqueeze(-1) + r1.unsqueeze(-1) * torch.cos(thetas),
                      y.unsqueeze(-1) + r2.unsqueeze(-1) * torch.sin(thetas)), dim=-1)

    return xy

def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: torch.device = None,
            dtype: torch.dtype = None,
            eps: float = 1e-6) -> torch.Tensor:
    r"""Converts an integer label 2D tensor to a one-hot 3D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch siz. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor.
    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> tgm.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 3:
        raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                         .format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.
    According to [1], we compute the Sørensen-Dice Coefficient as follows:
    .. math::
        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}
    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.
    the loss, is finally computed as:
    .. math::
        \text{loss}(x, class) = 1 - \text{Dice}(x, class)
    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, softmax=True) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.softmax = softmax

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, target.device))
        # compute softmax over the classes axis
        if self.softmax:
            input_soft = F.softmax(input, dim=1)
        else:
            input_soft = input

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)

def plot_loss(train_loss, val_loss, max_epochs, epoch, id_, y):

    plt.figure(figsize=(25,8))
    plt.plot(list(range(epoch+1)), train_loss, label='Train')
    plt.plot(list(range(epoch+1)), val_loss, label='Val')
    plt.yticks(np.arange(y[0], y[1], y[2]))
    plt.xticks(np.arange(0, max_epochs, 5))
    plt.grid(alpha=0.5)
    plt.xlim(0,max_epochs)
    plt.ylim(y[0],y[1])
    plt.legend()
    plt.title(id_)
    plt.savefig(id_, dpi=300)
    #plt.show()
    plt.close()
