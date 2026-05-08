from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics.utils.metrics import bbox_iou, bbox_nwd
from ultralytics.utils.loss import BboxLoss, bbox2dist


_AWL_ALPHA = 0.6
_AWL_ENABLED = True


def set_awl_ratio(alpha: float) -> None:
    global _AWL_ALPHA
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha 必须在 [0,1] 范围内，当前为: {alpha}")
    _AWL_ALPHA = float(alpha)


def get_awl_ratio() -> float:
    return _AWL_ALPHA


def enable_awl(enabled: bool = True) -> None:
    global _AWL_ENABLED
    _AWL_ENABLED = bool(enabled)


def _patched_bbox_loss_forward(
    self,
    pred_dist: torch.Tensor,
    pred_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    fg_mask: torch.Tensor,
    imgsz: torch.Tensor,
    stride: torch.Tensor,
):

    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

    if _AWL_ENABLED:
        nwd = bbox_nwd(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        alpha = _AWL_ALPHA
        fused_metric = alpha * iou + (1.0 - alpha) * nwd
        loss_iou = ((1.0 - fused_metric) * weight).sum() / target_scores_sum
    else:
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum


    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(
            pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
            target_ltrb[fg_mask]
        ) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        target_ltrb = bbox2dist(anchor_points, target_bboxes)
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]
        loss_dfl = (
            F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum

    return loss_iou, loss_dfl


def patch_awl(alpha: float = 0.6, verbose: bool = True) -> None:
    set_awl_ratio(alpha)
    enable_awl(True)
    BboxLoss.forward = _patched_bbox_loss_forward
    if verbose:
        print(f"[Patch] AWL enabled: alpha={alpha:.2f}, beta={1.0 - alpha:.2f}")


def disable_awl(verbose: bool = True) -> None:
    enable_awl(False)
    if verbose:
        print("[Patch] AWL disabled, fallback to pure CIoU")