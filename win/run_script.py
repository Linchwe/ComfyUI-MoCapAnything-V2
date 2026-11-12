"""원본 전처리 스크립트를 윈도우 패치(wincompat)를 적용한 상태로 실행하는 러너.

사용법: python win/run_script.py <스크립트 경로> [스크립트 인자...]

 - ComfyUI 포터블 파이썬(._pth 격리 모드)은 PYTHONPATH/현재 폴더를 무시하므로
   저장소 루트를 직접 sys.path 에 넣고 스크립트를 __main__ 으로 실행합니다.
"""
import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import wincompat  # noqa: E402,F401  (UTF-8, 스레드 풀 패치)

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(2)

script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
