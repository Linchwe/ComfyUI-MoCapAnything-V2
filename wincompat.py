"""Windows 호환 패치 (MocapAnything Win).

모든 진입점(앱, CLI, 전처리 러너) 맨 위에서 `import wincompat` 한 줄로 적용합니다.
리눅스/맥에서는 아무 것도 바꾸지 않습니다.

하는 일
 - 콘솔 출력 UTF-8 고정: 원본 코드의 이모지/중국어 print 가 cp949 콘솔에서 죽지 않게
 - open() 기본 인코딩 UTF-8: 설정 yaml 의 중국어 주석 등을 cp949 로 읽다 죽지 않게
 - ProcessPoolExecutor / multiprocessing.Pool → 스레드 풀:
   윈도우 spawn 방식에서 runpy 로 실행한 스크립트의 함수를 자식 프로세스가 못 찾는 문제 회피
   (캐릭터 1개 전처리라 작업량이 작아서 스레드로 충분)
 - 저장소 루트와 vendor(roma) 를 sys.path 에 추가: ComfyUI 포터블 python_embeded(._pth 격리 모드)는
   PYTHONPATH 와 현재 폴더를 무시하므로
 - 모델/캐릭터 데이터 경로(MODELS_DIR 등)를 한 곳에서 정함
"""
import builtins
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
IS_WIN = os.name == "nt"


def _default_models_dir():
    """ComfyUI/custom_nodes/<이 폴더> 에 있으면 ComfyUI/models/mocapanything, 아니면 이 폴더 안."""
    parent = os.path.dirname(REPO_ROOT)
    if os.path.basename(parent) == "custom_nodes":
        return os.path.join(os.path.dirname(parent), "models", "mocapanything")
    return os.path.join(REPO_ROOT, "_models")


# 모델·캐릭터 데이터 위치 (ComfyUI 노드가 MCA_MODELS_DIR 로 넘겨줌 — extra_model_paths 설정도 반영됨)
MODELS_DIR = os.environ.get("MCA_MODELS_DIR") or _default_models_dir()
CKPT_DIR = os.path.join(MODELS_DIR, "checkpoints")           # MoCapAnything V2 가중치
HF_DIR = os.path.join(MODELS_DIR, "hf")                       # DINOv2 / BiRefNet / SAM2 / T5 (HuggingFace 캐시)
WORKSPACE = os.environ.get("MCA_WORKSPACE") or os.path.join(MODELS_DIR, "characters")   # 준비한 내 캐릭터


def _is_comfy_root(d):
    try:
        return (os.path.isfile(os.path.join(d, "main.py")) and os.path.isfile(os.path.join(d, "folder_paths.py"))
                and os.path.isdir(os.path.join(d, "comfy")))
    except (TypeError, ValueError):
        return False


def _ensure_repo_on_path():
    # ComfyUI portable's python_embeded ._pth file puts the ComfyUI folder on sys.path. ComfyUI has its own
    # `utils` package, which would shadow this repo's utils/models/... in the compute subprocess
    # ("No module named 'utils.bvh'"). The subprocess needs nothing from ComfyUI, so drop that folder.
    sys.path[:] = [p for p in sys.path if not _is_comfy_root(p or os.getcwd())]
    for name in [m for m in list(sys.modules) if m in ("utils", "models", "data") or m.startswith(("utils.", "models.", "data."))]:
        mod_file = getattr(sys.modules[name], "__file__", None) or ""
        if mod_file and not os.path.abspath(mod_file).startswith(REPO_ROOT):
            del sys.modules[name]
    if REPO_ROOT in sys.path:
        sys.path.remove(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)
    vendor = os.path.join(REPO_ROOT, "vendor")   # roma (BSD-3, NAVER) — pip 설치 없이 쓰도록 동봉
    if vendor not in sys.path:
        sys.path.append(vendor)                  # 사용자가 따로 설치한 roma 가 있으면 그쪽이 우선


def _utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _utf8_open_default():
    if getattr(builtins, "_mca_open_patched", False):
        return
    _orig_open = builtins.open

    def _open(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
        if encoding is None and "b" not in mode:
            encoding = "utf-8"
        return _orig_open(file, mode, buffering, encoding, *args, **kwargs)

    builtins.open = _open
    builtins._mca_open_patched = True


def _threads_instead_of_processes():
    import concurrent.futures as cf
    import multiprocessing
    import multiprocessing.pool as mpp

    cf.ProcessPoolExecutor = cf.ThreadPoolExecutor
    multiprocessing.Pool = mpp.ThreadPool


def to_posix(path):
    """경로 문자열의 역슬래시를 슬래시로 (원본 코드의 split('/') 대응용)."""
    return path.replace("\\", "/") if isinstance(path, str) else path


_ensure_repo_on_path()
_utf8_stdio()
# HuggingFace 모델을 C 드라이브 사용자 폴더가 아니라 ComfyUI/models/mocapanything/hf 에 받음
os.environ["HF_HOME"] = HF_DIR
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # 노드 로그가 진행바로 도배되지 않게
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without Developer Mode: files are copied instead (expected)
if IS_WIN:
    _utf8_open_default()
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if os.environ.get("MCA_ALLOW_PROCESS_POOL") != "1":
        _threads_instead_of_processes()


def blender_args():
    """블렌더 실행 시 `-b` 바로 뒤에 넣을 인자.

    - --factory-startup : 사용자 애드온(MCP 서버 등)이 백그라운드 실행을 방해하지 않게
    - --python-expr     : 블렌더 내장 파이썬에도 같은 UTF-8 패치 적용
                          (블렌더는 PYTHONIOENCODING 같은 환경변수를 무시함)
    """
    expr = f"import sys; sys.path.insert(0, {REPO_ROOT!r}); import wincompat"
    return ["--factory-startup", "--python-expr", expr]
