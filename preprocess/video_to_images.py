import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

from tqdm import tqdm


_zoo = os.environ.get('ZOO_ROOT', 'zoo')   # MocapAnything Win: ZOO_ROOT 지원
video_root = os.path.join(_zoo, 'video')
image_root = os.path.join(_zoo, 'image')
fps = 30


def extract_frames(video_path):
    motion_name = os.path.basename(os.path.dirname(video_path))
    view = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(image_root, motion_name, view)
    os.makedirs(out_dir, exist_ok=True)

    try:  # MocapAnything Win: 시스템 ffmpeg 대신 imageio-ffmpeg 내장 바이너리
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = 'ffmpeg'
    cmd = [
        ffmpeg_exe, '-y', '-loglevel', 'error',
        '-i', video_path,
        '-vf', f'fps={fps}',
        '-start_number', '0',
        os.path.join(out_dir, '%05d.png'),
    ]
    subprocess.run(cmd, check=True)
    return f"{motion_name}/{view}"


def main(num_workers=16):
    videos = sorted(glob(os.path.join(video_root, '*', '*.mp4')))
    print(f"==> Extracting {len(videos)} videos at {fps} fps with {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(extract_frames, v) for v in videos]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="ffmpeg"):
            try:
                fut.result()
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] {e}")

    print("==> Done.")


if __name__ == "__main__":
    main(num_workers=int(os.environ.get('MCA_WORKERS', '4')))
