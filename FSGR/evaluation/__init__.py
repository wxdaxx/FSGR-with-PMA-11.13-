# evaluation/__init__.py
from .bleu import Bleu
from .rouge import Rouge
from .cider import Cider
from .tokenizer import PTBTokenizer

import os

# 可选启用：需要 Java 环境
_USE_METEOR = os.getenv("FSGR_USE_METEOR", "0") == "1"
_USE_SPICE  = os.getenv("FSGR_USE_SPICE",  "0") == "1"

Meteor = None
Spice = None

if _USE_METEOR:
    try:
        from .meteor import Meteor  # noqa: F401
    except Exception:
        Meteor = None

if _USE_SPICE:
    try:
        from .spice import Spice  # noqa: F401
    except Exception:
        Spice = None


def compute_scores(gts, gen):
    """
    默认只跑 BLEU/ROUGE/CIDEr，稳定且无需 Java。
    若设置环境变量 FSGR_USE_METEOR=1 或 FSGR_USE_SPICE=1 且环境满足依赖，则会额外计算。
    """
    metrics = [Bleu(), Rouge(), Cider()]
    if _USE_METEOR and Meteor is not None:
        metrics.append(Meteor())
    if _USE_SPICE and Spice is not None:
        metrics.append(Spice())

    all_score, all_scores = {}, {}
    for metric in metrics:
        try:
            score, scores = metric.compute_score(gts, gen)
        except Exception:
            # 保底：单个指标失败不影响整体
            score, scores = 0.0, []
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores
    return all_score, all_scores


__all__ = ['Bleu', 'Rouge', 'Cider', 'PTBTokenizer']
