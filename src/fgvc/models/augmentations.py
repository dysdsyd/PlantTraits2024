import torch
import torch.nn as nn
import math
from timm.data.mixup import Mixup, cutmix_bbox_and_lam, one_hot


class Mixup_transmix(Mixup):
    """act like Mixup(), but return useful information with method transmix_label()
        Mixup/Cutmix that applies different params to each element or whole batch, where per-batch is set as default

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
        transmix (bool): enable TransMix or not
    """

    def __init__(
        self,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        correct_lam=True,
        label_smoothing=0.1,
        num_classes=17396,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = (
            correct_lam  # correct lambda based on clipped area for cutmix
        )
        self.mixup_enabled = (
            True  # set to false to disable mixing (intended tp be set by train loop)
        )

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()

        if lam == 1.0:
            return 1.0
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape,
                lam,
                ratio_minmax=self.cutmix_minmax,
                correct_lam=self.correct_lam,
            )
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]  # cutmix for input!
            return lam, (yl, yh, xl, xh)  # return box!
        else:
            x_flipped = x.flip(0).mul_(1.0 - lam)
            x.mul_(lam).add_(x_flipped)

        return lam

    def transmix_label(self, target, attn, input_shape, ratio=0.5):
        """use the self information?
        args:
            attn (torch.tensor): attention map from the last Transformer with shape (N, hw)
            target (tuple): (target, y1, y2, use_cutmix, box)
                target (torch.tensor): mixed target by area-ratio
                y1 (torch.tensor): one-hot label for image A (background image) (N, k)
                y2 (torch.tensor): one-hot label for image B (cropped patch)  (N, k)
                use_cutmix (bool): enable cutmix if True, otherwise enable Mixup
                box (tuple): (yl, yh, xl, xh)
        returns:
            target (torch.tensor): with shape (N, K)
        """
        # the placeholder _ is the area-based target
        (_, y1, y2, box) = target
        lam0 = (box[1] - box[0]) * (box[3] - box[2]) / (input_shape[2] * input_shape[3])
        mask = torch.zeros((input_shape[2], input_shape[3]))
        mask[box[0] : box[1], box[2] : box[3]] = 1
        mask = nn.Upsample(size=int(math.sqrt(attn.shape[1])))(
            mask.unsqueeze(0).unsqueeze(0)
        ).int()
        mask = mask.view(1, -1).repeat(len(attn), 1)  # (b, hw)
        w1, w2 = torch.sum((1 - mask) * attn, dim=1), torch.sum(mask * attn, dim=1)
        lam1 = w2 / (w1 + w2)  # (b, )
        lam = (lam0 + lam1) / 2  # ()+(b,) ratio=0.5
        target = y1 * (1.0 - lam).unsqueeze(1) + y2 * lam.unsqueeze(1)
        return target

    def __call__(self, x, x_tab, y_reg, y_clf):
        assert len(x) % 2 == 0, "Batch size should be even when using this"
        assert self.mode == "batch", "Mixup mode is batch by default"
        lam = self._mix_batch(x)  # tuple or value
        if isinstance(lam, tuple):
            lam, box = lam  # lam: (b,)
            use_cutmix = True
        else:  # lam is a value
            use_cutmix = False

        x_tab, y_reg, y_clf = mixup_target(
            x_tab,
            y_reg,
            y_clf,
            self.num_classes,
            lam,
            self.label_smoothing,
        )
        return x, x_tab, y_reg, y_clf, lam


def mixup_target(x_tab, y_reg, y_clf, num_classes, lam=1.0, smoothing=0.0):
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y1 = one_hot(y_clf, num_classes, on_value=on_value, off_value=off_value)
    y2 = one_hot(
        y_clf.flip(0),
        num_classes,
        on_value=on_value,
        off_value=off_value,
    )
    y_clf = y1 * lam + y2 * (1.0 - lam)
    y_reg = y_reg * lam + y_reg.flip(0) * (1.0 - lam)
    x_tab = x_tab * lam + x_tab.flip(0) * (1.0 - lam)
    return x_tab, y_reg, y_clf
