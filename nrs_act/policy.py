import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

# detr가 없는 환경에서도 import 에러로 바로 죽지 않게 처리
try:
    from detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
except ImportError as e:
    # 이 메시지는 import 시점에 한 번만 뜨도록
    raise ImportError(
        "[policy.py] 'detr' 패키지를 찾을 수 없습니다. "
        "원래 ACT 레포의 'detr/' 디렉토리를 현재 프로젝트(root)에 그대로 옮겨놓거나, "
        "PYTHONPATH에 추가해주세요.\n"
        f"원래 에러: {e}"
    )

import IPython
e = IPython.embed


class ACTPolicy(nn.Module):
    """
    - 학습 시: __call__(qpos, image, actions, is_pad) -> loss_dict
    - 추론 시: __call__(qpos, image) -> 예측 action 시퀀스
    detr.main 쪽에서 실제 비전+Transformer 모델을 만들어서 여기서 한 번 감싸는 구조.
    """

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(f"[ACTPolicy] KL Weight = {self.kl_weight}")

        # 이미지 정규화는 여기서 공통으로
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 값
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, qpos, image, actions=None, is_pad=None):
        """
        qpos: (B, T_qpos, Dq) 또는 (B, Dq)  - detr 모델 내부에서 기대하는 shape는 detr 쪽 구현을 따름
        image: (B, K, C, H, W)  # K = camera 개수
        actions: (B, T_action, Da)
        is_pad: (B, T_action)
        """
        env_state = None  # 지금은 안 씀

        # image가 float여야 하고, 카메라 차원까지 들어온 상태에서 normalize
        # image: (B, K, C, H, W)
        image = self._normalize(image)

        # ===== 학습 모드 =====
        if actions is not None:
            # 모델이 예측하는 길이(num_queries)까지만 자른다
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            # 모델 forward
            # a_hat: (B, T, Da)
            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad
            )

            # KL
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            # L1 loss, padding 된 구간은 빼고 평균
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")  # (B, T, Da)
            # is_pad: (B, T) -> (B, T, 1)로 늘려서 브로드캐스트
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {
                "l1": l1,
                "kl": total_kld[0],
            }
            loss_dict["loss"] = loss_dict["l1"] + self.kl_weight * loss_dict["kl"]
            return loss_dict

        # ===== 추론 모드 =====
        else:
            # actions 안 주면 모델이 prior에서 샘플해서 시퀀스를 뽑는다
            a_hat, _, _ = self.model(qpos, image, env_state)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    """
    더 단순한 CNN+MLP 정책. 카메라 한 장(or 여러 장 묶음) + qpos -> 단일 action.
    """

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

        # ===== 학습 =====
        if actions is not None:
            # CNNMLP는 한 스텝만 본다고 가정
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            return {"mse": mse, "loss": mse}

        # ===== 추론 =====
        else:
            a_hat = self.model(qpos, image, env_state)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    """
    원본 ACT가 쓰는 KL 계산 그대로 유지
    """
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
