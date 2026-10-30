"""Copy source simulation videos into the static site and update its media list.

Run with Python from any directory. OpenCV is optional: when installed it also
extracts original-color poster frames; otherwise existing posters are reused.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'videos'
    destination = root / 'docs/assets/videos'
    config_path = root / 'docs/assets/site-config.js'
    match = re.search(r'window\.FACA_SITE\s*=\s*(\{.*\})\s*;\s*$',
                      config_path.read_text(), re.S)
    if match is None:
        raise ValueError('Expected a JSON object assigned to window.FACA_SITE')
    config = json.loads(match[1])
    previous = {Path(item.get('src', '')).name: item for item in config.get('videos', [])}
    try:
        import cv2
    except ImportError:
        cv2 = None
    files = sorted(path for path in source.iterdir()
                   if path.is_file() and path.suffix.lower() in ('.mp4', '.webm'))
    if not files:
        raise ValueError(f'No video files found in {source}')
    destination.mkdir(parents=True, exist_ok=True)
    videos = []
    for index, video_path in enumerate(files):
        target = destination / video_path.name
        if (not target.exists() or target.stat().st_size != video_path.stat().st_size
                or target.stat().st_mtime_ns != video_path.stat().st_mtime_ns):
            shutil.copy2(video_path, target)
        poster = destination / f'{video_path.stem}-poster.jpg'
        if cv2 is not None and (not poster.exists()
                               or poster.stat().st_mtime_ns < video_path.stat().st_mtime_ns):
            capture = cv2.VideoCapture(str(video_path))
            try:
                if capture.isOpened():
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, min(int(fps), max(int(frames)-1, 0)))
                    success, frame = capture.read()
                    if success:
                        cv2.imwrite(str(poster), frame)
            finally:
                capture.release()
        name = re.fullmatch(r'(\d+)_(\d+)', video_path.stem)
        default_title = (f'Assembly {name[1]} · Clip {name[2]}' if name else video_path.stem)
        old = previous.get(video_path.name, {})
        videos.append({
            'id': 'overview' if index == 0 else video_path.stem,
            'title': old.get('title', default_title),
            'src': f'assets/videos/{video_path.name}',
            'poster': f'assets/videos/{poster.name}' if poster.exists() else '',
            'captions': old.get('captions', ''),
            **{key: old[key] for key in ('captionsLabel', 'captionsLanguage') if key in old},
            'type': 'video/webm' if video_path.suffix.lower() == '.webm' else 'video/mp4',
        })
    config['videos'] = videos
    header = ('/* Paths are relative to docs/index.html. The overview entry is the main video;\n'
              ' * other entries populate the simulation gallery.\n'
              ' * Refresh from videos/: python scripts/sync_site_videos.py\n'
              ' */\nwindow.FACA_SITE = ')
    config_path.write_text(header + json.dumps(config, ensure_ascii=False, indent=2) + ';\n')
    print(f'Synchronized {len(videos)} videos: 1 featured and {len(videos)-1} gallery clips.')
    if cv2 is None:
        print('OpenCV not installed: existing posters retained; new clips use native video previews.')


if __name__ == '__main__':
    main()
