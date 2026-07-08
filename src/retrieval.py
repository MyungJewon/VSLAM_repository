# 전역 이미지 검색(place recognition) — 이미지 1장 → 고정 길이 벡터.
# DB 벡터들과 코사인 유사도로 top-k 후보를 고른다 (NetVLAD 역할).
import cv2
import numpy as np
import torch
import torchvision.transforms as T


class Retrieval:
    """model_name: 'megaloc'(정확, 무거움) | 'cosplace'(경량 ResNet18 폴백)"""

    def __init__(self, model_name: str = 'megaloc'):
        if model_name == 'megaloc':
            self.model = torch.hub.load('gmberton/MegaLoc', 'get_trained_model',
                                        trust_repo=True)
        else:
            self.model = torch.hub.load('gmberton/cosplace', 'get_trained_model',
                                        backbone='ResNet18', fc_output_dim=512,
                                        trust_repo=True)
        self.model.eval()
        self.tf = T.Compose([
            T.ToTensor(),
            T.Resize((322, 322), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        """BGR 이미지 → L2 정규화된 디스크립터 벡터 (1D)"""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        v = self.model(self.tf(rgb)[None]).squeeze(0).cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-12)


def top_k(query_vec: np.ndarray, db_vecs: np.ndarray, k: int = 5) -> np.ndarray:
    """코사인 유사도 상위 k개 DB 인덱스 (유사도 내림차순)."""
    sims = db_vecs @ query_vec
    return np.argsort(-sims)[:k]
