"""Visual Semantic Engine의 index/search 명령행 인터페이스입니다."""

from __future__ import annotations

import argparse
from pathlib import Path

from .encoder import OpenClipSemanticEncoder
from .index import SemanticIndex


# 일반적인 이미지 파일만 자동 index 대상으로 인정합니다.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(directory: Path) -> list[Path]:
    """하위 폴더까지 재귀적으로 탐색해 지원 이미지 경로를 정렬해서 반환합니다."""

    # 실행마다 동일한 파일 순서를 얻도록 resolve 후 문자열 기준으로 정렬합니다.
    return sorted(
        [path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: str(path.resolve()),
    )


def build_index(args: argparse.Namespace) -> None:
    """이미지 디렉터리를 OpenCLIP 임베딩 index로 변환합니다."""

    # 입력 폴더에서 실제 처리할 이미지 파일 목록을 수집합니다.
    image_paths = collect_images(args.directory)
    # 비어 있는 폴더는 조용히 성공시키지 않고 사용자에게 바로 알려줍니다.
    if not image_paths:
        raise SystemExit(f"no supported images found under: {args.directory}")

    # 모델 loading과 preprocessing은 OpenCLIP 원본 factory를 감싼 얇은 encoder에 위임합니다.
    encoder = OpenClipSemanticEncoder(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )
    # 이미지 전체를 batch 단위로 임베딩합니다.
    embeddings = encoder.encode_images(image_paths, batch_size=args.batch_size)
    # 파일 경로와 모델 metadata를 하나의 persistent index 객체로 묶습니다.
    index = SemanticIndex.from_tensor(
        paths=image_paths,
        embeddings=embeddings,
        model_name=encoder.model_name,
        pretrained=encoder.pretrained,
    )
    # 이후 모델을 다시 돌리지 않고 검색할 수 있도록 로컬 disk에 저장합니다.
    index.save(args.output)
    # 자동화 스크립트에서도 읽기 쉬운 간단한 완료 메시지를 출력합니다.
    print(f"indexed {len(image_paths)} images -> {args.output}")


def search_index(args: argparse.Namespace) -> None:
    """저장된 index를 자연어 또는 이미지 query로 검색합니다."""

    # 먼저 index metadata를 읽어 생성 당시 사용했던 모델을 확인합니다.
    index = SemanticIndex.load(args.index)
    # index와 다른 모델 공간을 사용하면 similarity가 무의미하므로 동일 설정으로 encoder를 만듭니다.
    encoder = OpenClipSemanticEncoder(
        model_name=index.model_name,
        pretrained=index.pretrained,
        device=args.device,
    )

    # 텍스트 query가 주어지면 text encoder를 사용합니다.
    if args.text is not None:
        query_embedding = encoder.encode_texts([args.text])[0]
    else:
        # 이미지 query가 주어지면 index와 동일한 vision encoder를 사용합니다.
        query_embedding = encoder.encode_images([args.image], batch_size=1)[0]

    # 의미 유사도 상위 결과를 가져옵니다.
    results = index.search(query_embedding, top_k=args.top_k)
    # 다른 shell 도구와 연결하기 쉽도록 한 줄에 score와 path를 tab으로 출력합니다.
    for result in results:
        print(f"{result.score:.6f}\t{result.path}")


def create_parser() -> argparse.ArgumentParser:
    """index와 search 하위 명령을 포함한 CLI parser를 생성합니다."""

    # 최상위 프로그램 설명을 정의합니다.
    parser = argparse.ArgumentParser(prog="visual-semantic")
    # index/search처럼 역할이 다른 명령을 subcommand로 구분합니다.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 이미지 디렉터리를 embedding index로 만드는 명령을 정의합니다.
    index_parser = subparsers.add_parser("index", help="build a semantic image index")
    # 검색 대상 이미지가 들어 있는 루트 폴더를 필수 인자로 받습니다.
    index_parser.add_argument("directory", type=Path)
    # index 저장 디렉터리를 필수 옵션으로 받습니다.
    index_parser.add_argument("--output", type=Path, required=True)
    # OpenCLIP model architecture를 필요에 따라 교체할 수 있도록 노출합니다.
    index_parser.add_argument("--model", default="ViT-B-32")
    # 기본값은 OpenAI 원본 CLIP pretrained weight를 사용합니다.
    index_parser.add_argument("--pretrained", default="openai")
    # 장치를 명시하지 않으면 encoder가 CUDA/MPS/CPU를 자동 선택합니다.
    index_parser.add_argument("--device", default=None)
    # 메모리 상황에 따라 inference batch 크기를 조정할 수 있게 합니다.
    index_parser.add_argument("--batch-size", type=int, default=32)
    # 해당 subcommand 실행 함수를 parser에 연결합니다.
    index_parser.set_defaults(handler=build_index)

    # 이미 생성한 index를 검색하는 명령을 정의합니다.
    search_parser = subparsers.add_parser("search", help="search an existing semantic index")
    # 검색할 index 디렉터리를 위치 인자로 받습니다.
    search_parser.add_argument("index", type=Path)
    # 텍스트와 이미지 query는 동시에 지정하지 못하도록 mutually exclusive group을 사용합니다.
    query_group = search_parser.add_mutually_exclusive_group(required=True)
    # 자연어 검색 query를 지원합니다.
    query_group.add_argument("--text")
    # 이미지 유사도 검색 query를 지원합니다.
    query_group.add_argument("--image", type=Path)
    # 반환할 결과 개수를 지정합니다.
    search_parser.add_argument("--top-k", type=int, default=10)
    # 검색 시에도 원하는 실행 장치를 직접 선택할 수 있습니다.
    search_parser.add_argument("--device", default=None)
    # 해당 subcommand 실행 함수를 parser에 연결합니다.
    search_parser.set_defaults(handler=search_index)

    # 완성된 parser를 호출자에게 반환합니다.
    return parser


def main() -> None:
    """CLI entry point입니다."""

    # 명령행 인자를 정의한 parser로 현재 argv를 해석합니다.
    args = create_parser().parse_args()
    # subcommand가 등록한 실제 실행 함수를 호출합니다.
    args.handler(args)


if __name__ == "__main__":
    # `python -m visual_semantic_engine.cli` 실행도 지원합니다.
    main()

