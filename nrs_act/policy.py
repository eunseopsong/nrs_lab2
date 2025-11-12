import sys, os
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

# -----------------------------
# 경로 보정: detr가 nrs_act/act/detr 에 있을 때 자동 인식되도록 추가
# -----------------------------
current_dir = os.path.dirname(__file__)
act_dir = os.path.join(current_dir, "act")
detr_dir = os.path.join(act_dir, "detr")

for path in [act_dir, detr_dir]:
    if path not in sys.path:
        sys.path.append(path)

# detr import (경로 보정 이후 시도)
try:
    from detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
except ImportError as e:
    raise ImportError(
        f"[policy.py] 'detr' 패키지를 찾을 수 없습니다.\n"
        f"현재 경로: {os.getcwd()}\n"
        f"시도된 경로: {detr_dir}\n"
        f"해결 방법: 'act/detr' 폴더가 존재하는지 확인하거나, PYTHONPATH에 추가하세요.\n"
        f"원래 에러: {e}"
    )

import IPython
e = IPython.embed


class ACTPolicy(nn.Module):
    """ACT 기반 정책 네트워크"""

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(f"[ACTPolicy] KL Weight = {self.kl_weight}")

        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        image = self._normalize(image)

        # 학습 모드
        if actions is not None:
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad
            )

            total_kld, _, _ = kl_divergence(mu, logvar)
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {"l1": l1, "kl": total_kld[0]}
            loss_dict["loss"] = loss_dict["l1"] + self.kl_weight * loss_dict["kl"]
            return loss_dict

        # 추론 모드
        else:
            a_hat, _, _ = self.model(qpos, image, env_state)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    """단순 CNN+MLP 정책"""

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        image = self._normalize(image)

        if actions is not None:
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            return {"mse": mse, "loss": mse}
        else:
            a_hat = self.model(qpos, image, env_state)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    """ACT에서 사용하는 KL 계산"""
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
