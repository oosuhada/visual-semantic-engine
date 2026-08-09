"""임베딩을 파일 시스템에 저장하고 검색하는 가벼운 persistent index입니다."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SearchResult:
    """검색 결과 한 건의 파일 경로와 의미 유사도 점수를 표현합니다."""

    path: str
    score: float


class SemanticIndex:
    """정규화된 임베딩 행렬과 원본 파일 경로를 함께 보관하는 로컬 index입니다."""

    def __init__(
        self,
        paths: list[str],
        embeddings: np.ndarray,
        model_name: str,
        pretrained: str,
    ) -> None:
        # 검색 결과가 원본 파일과 정확히 연결되도록 경로 순서를 그대로 보관합니다.
        self.paths = paths
        # dot product 검색을 위해 float32 임베딩 행렬을 보관합니다.
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        # index가 어떤 모델로 생성됐는지 확인할 수 있도록 모델 이름을 저장합니다.
        self.model_name = model_name
        # 사용한 pretrained checkpoint tag도 함께 저장합니다.
        self.pretrained = pretrained

        # 파일 수와 임베딩 row 수가 다르면 잘못된 index이므로 즉시 실패시킵니다.
        if len(self.paths) != self.embeddings.shape[0]:
            raise ValueError("number of paths must match number of embeddings")
        # 검색 행렬은 반드시 [N, D] 2차원이어야 합니다.
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D matrix")

    @classmethod
    def from_tensor(
        cls,
        paths: list[Path],
        embeddings: torch.Tensor,
        model_name: str,
        pretrained: str,
    ) -> "SemanticIndex":
        # 모델 Tensor 결과를 NumPy index로 변환해 저장 계층과 모델 계층을 분리합니다.
        return cls(
            paths=[str(path.resolve()) for path in paths],
            embeddings=embeddings.detach().cpu().numpy(),
            model_name=model_name,
            pretrained=pretrained,
        )

    def save(self, directory: Path) -> None:
        """index를 사람이 읽을 수 있는 metadata와 빠른 NumPy 행렬로 나누어 저장합니다."""

        # 사용자가 지정한 index 디렉터리가 없으면 자동 생성합니다.
        directory.mkdir(parents=True, exist_ok=True)
        # 큰 숫자 행렬은 JSON보다 효율적인 NumPy binary 형식으로 저장합니다.
        np.save(directory / "embeddings.npy", self.embeddings)
        # 경로와 모델 정보는 Git diff나 수동 확인이 쉬운 JSON으로 저장합니다.
        metadata = {
            "format": "visual-semantic-index-v1",
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "paths": self.paths,
        }
        # 한글 파일명도 손실 없이 유지하도록 ensure_ascii를 끕니다.
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "SemanticIndex":
        """저장된 index를 다시 메모리로 불러옵니다."""

        # metadata JSON을 먼저 읽어 index 형식과 모델 정보를 복원합니다.
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        # 다른 형식의 파일을 실수로 읽지 않도록 format version을 확인합니다.
        if metadata.get("format") != "visual-semantic-index-v1":
            raise ValueError("unsupported semantic index format")
        # 임베딩 행렬은 pickle을 허용하지 않는 안전한 NumPy 로딩으로 읽습니다.
        embeddings = np.load(directory / "embeddings.npy", allow_pickle=False)
        # 복원한 값으로 동일한 SemanticIndex 객체를 생성합니다.
        return cls(
            paths=list(metadata["paths"]),
            embeddings=embeddings,
            model_name=str(metadata["model_name"]),
            pretrained=str(metadata["pretrained"]),
        )

    def search(self, query_embedding: np.ndarray | torch.Tensor, top_k: int = 10) -> list[SearchResult]:
        """정규화된 query embedding과 가장 가까운 파일을 cosine similarity 순으로 반환합니다."""

        # PyTorch Tensor가 들어오면 저장 계층에서 사용할 NumPy 배열로 변환합니다.
        if isinstance(query_embedding, torch.Tensor):
            query = query_embedding.detach().cpu().numpy()
        else:
            query = np.asarray(query_embedding, dtype=np.float32)

        # batch 차원이 하나인 경우 첫 번째 query만 사용하도록 [D] 벡터로 평탄화합니다.
        query = query.reshape(-1).astype(np.float32)
        # 외부 encoder가 정규화를 생략했더라도 안전하게 query를 다시 정규화합니다.
        query_norm = np.linalg.norm(query)
        if query_norm <= 1e-12:
            raise ValueError("query embedding must have non-zero norm")
        query = query / query_norm

        # 저장된 embedding도 혹시 모를 부동소수점 오차를 보정해 재정규화합니다.
        row_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized = self.embeddings / np.clip(row_norms, 1e-12, None)
        # cosine similarity는 정규화된 벡터 간 dot product로 계산합니다.
        scores = normalized @ query
        # 요청한 top_k가 전체 파일 수보다 클 경우 가능한 범위로 제한합니다.
        limit = min(max(top_k, 1), len(self.paths))
        # 전체 정렬 대신 argpartition으로 후보를 먼저 줄여 큰 index에서도 비용을 낮춥니다.
        candidate_indices = np.argpartition(-scores, limit - 1)[:limit]
        # 최종 후보만 실제 점수 기준으로 정렬합니다.
        ordered_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
        # UI나 CLI에서 바로 사용할 수 있도록 파일 경로와 float 점수로 변환합니다.
        return [
            SearchResult(path=self.paths[index], score=float(scores[index]))
            for index in ordered_indices
        ]

