"""MocapAnything Win — 배경 제거기 진입점.

원본 저장소는 이 파일에 BRIA RMBG-1.4 네트워크 코드를 담고 있었고,
RMBG-1.4 는 BRIA 비상업 라이선스라 상업 작업에 쓸 수 없습니다.
원본 코드 전체가 `from preprocess.briarmbg import BriaRMBG` 후
`BriaRMBG.from_pretrained(...)` 로 배경 제거기를 만들기 때문에,
이름만 그대로 두고 MIT 라이선스 모델(BiRefNet / InSPyReNet)로 바꿔 끼웠습니다.
BRIA 코드는 이 파일에서 완전히 제거했습니다.
"""
try:
    from preprocess.mit_matting import MitMatting as BriaRMBG
except ImportError:  # preprocess 폴더 안에서 직접 실행한 경우
    from mit_matting import MitMatting as BriaRMBG

__all__ = ["BriaRMBG"]
