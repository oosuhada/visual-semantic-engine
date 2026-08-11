from pathlib import Path

import numpy as np

from visual_semantic_engine.index import SemanticIndex


def test_semantic_index_round_trip_and_search(tmp_path: Path) -> None:
    # 세 파일이 서로 직교하는 toy embedding을 가진 상황을 준비합니다.
    index = SemanticIndex(
        paths=["a.jpg", "b.jpg", "c.jpg"],
        embeddings=np.eye(3, dtype=np.float32),
        model_name="toy-model",
        pretrained="toy-weights",
    )
    # 실제 index 파일 저장 경로를 테스트 임시 폴더 아래에 만듭니다.
    output = tmp_path / "index"
    # metadata와 embedding 행렬을 실제 disk에 저장합니다.
    index.save(output)
    # 저장된 파일을 다시 읽어 persistence가 동작하는지 검증합니다.
    restored = SemanticIndex.load(output)
    # 두 번째 축과 같은 query를 넣으면 b.jpg가 첫 번째 결과여야 합니다.
    results = restored.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), top_k=2)
    # 최상위 파일이 의도한 semantic neighbor인지 확인합니다.
    assert results[0].path == "b.jpg"
    # 완전히 같은 방향의 cosine similarity는 1에 가까워야 합니다.
    assert results[0].score == 1.0

