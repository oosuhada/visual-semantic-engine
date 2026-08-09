"""OpenCLIP 원본 구현을 그대로 호출하는 의미 임베딩 인코더입니다."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

import open_clip


class OpenClipSemanticEncoder:
    """OpenCLIP 모델을 이용해 이미지와 텍스트를 동일한 임베딩 공간으로 변환합니다."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
    ) -> None:
        # CUDA, Apple Silicon MPS, CPU 순서로 사용 가능한 장치를 자동 선택합니다.
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # 이후 encode 단계에서 동일 장치를 재사용하기 위해 장치 문자열을 보관합니다.
        self.device = device
        # 인덱스 재현성을 위해 실제 사용 모델 이름을 보관합니다.
        self.model_name = model_name
        # OpenAI 원본 가중치 또는 OpenCLIP 공개 가중치 tag를 그대로 보관합니다.
        self.pretrained = pretrained

        # 모델, 학습용 transform, 평가용 transform은 OpenCLIP factory에서 그대로 생성합니다.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        # 검색은 추론 작업이므로 dropout과 stochastic layer를 끄기 위해 eval 모드로 전환합니다.
        self.model.eval()
        # 텍스트 tokenizer 역시 OpenCLIP factory에서 직접 가져와 별도 구현을 만들지 않습니다.
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @staticmethod
    def _normalize(embeddings: torch.Tensor) -> torch.Tensor:
        # cosine similarity를 dot product로 빠르게 계산할 수 있도록 L2 정규화합니다.
        return embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode_images(self, image_paths: list[Path], batch_size: int = 32) -> torch.Tensor:
        """이미지 파일 경로 목록을 정규화된 CPU 임베딩 Tensor로 변환합니다."""

        # 최종 임베딩 batch들을 CPU에서 합치기 위한 리스트를 준비합니다.
        encoded_batches: list[torch.Tensor] = []

        # 모델 메모리 사용량을 제어하기 위해 지정한 batch 크기로 이미지를 나눕니다.
        for start in range(0, len(image_paths), batch_size):
            # 현재 batch에 해당하는 파일 경로만 선택합니다.
            batch_paths = image_paths[start : start + batch_size]
            # OpenCLIP 공식 preprocess를 각 PIL 이미지에 적용합니다.
            batch = torch.stack(
                [self.preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
            ).to(self.device)

            # inference에서는 gradient가 필요 없으므로 계산 그래프 생성을 끕니다.
            with torch.inference_mode():
                # 실제 vision encoder는 OpenCLIP 원본 구현을 그대로 호출합니다.
                embeddings = self.model.encode_image(batch)
                # 검색용 cosine similarity 계산을 위해 임베딩을 정규화합니다.
                embeddings = self._normalize(embeddings.float())

            # GPU/MPS 메모리를 계속 점유하지 않도록 결과를 즉시 CPU로 이동합니다.
            encoded_batches.append(embeddings.cpu())

        # 입력 이미지가 하나도 없으면 명확한 오류를 반환합니다.
        if not encoded_batches:
            raise ValueError("at least one image path is required")

        # 모든 batch를 하나의 [N, D] 임베딩 행렬로 합칩니다.
        return torch.cat(encoded_batches, dim=0)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """자연어 질의를 정규화된 CPU 임베딩 Tensor로 변환합니다."""

        # OpenCLIP tokenizer를 그대로 사용해 모델 입력 token을 생성합니다.
        tokens = self.tokenizer(texts).to(self.device)
        # 텍스트 검색 역시 inference만 필요하므로 gradient 계산을 끕니다.
        with torch.inference_mode():
            # 실제 text encoder는 OpenCLIP 원본 구현을 그대로 호출합니다.
            embeddings = self.model.encode_text(tokens)
            # 이미지 임베딩과 같은 방식으로 cosine 검색용 정규화를 수행합니다.
            embeddings = self._normalize(embeddings.float())
        # 검색 index는 CPU에서 관리하므로 결과를 CPU로 이동합니다.
        return embeddings.cpu()

