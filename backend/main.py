"""
FastAPI wrapper around the lane-detection pipeline (model/lane_detection.ipynb).

One endpoint: upload a dashcam video, get back the same clip (first CLIP_SECONDS
seconds) with the BEV lane overlay burned in. Processing is synchronous — native
OpenCV is fast enough on short clips that no job queue/polling is needed.
"""

import os
import subprocess
import tempfile
import uuid

import imageio_ffmpeg
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from lane_detection import process_video

MAX_BYTES = 100 * 1024 * 1024
CLIP_SECONDS = 30
MODES = {"auto", "day", "night"}

app = FastAPI(title="Lane Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["POST"],
    allow_headers=["*"],
)


def _cleanup(*paths: str) -> None:
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _transcode_to_h264(src_path: str, dst_path: str) -> None:
    """OpenCV's mp4v output isn't reliably playable in browsers; re-encode to
    H.264 with the static ffmpeg binary bundled by imageio-ffmpeg."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-y", "-i", src_path, "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed: {result.stderr[-1500:]}")


@app.post("/api/process")
async def process(background_tasks: BackgroundTasks, video: UploadFile, mode: str = "auto"):
    mode = mode.lower()
    if mode not in MODES:
        raise HTTPException(400, "mode must be one of: auto, day, night")
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(400, "Please upload a video file")

    data = await video.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Video exceeds the 100MB limit")

    job_id = uuid.uuid4().hex
    tmp_dir = tempfile.gettempdir()
    input_path = os.path.join(tmp_dir, f"lane-in-{job_id}.mp4")
    raw_output_path = os.path.join(tmp_dir, f"lane-raw-{job_id}.mp4")
    output_path = os.path.join(tmp_dir, f"lane-out-{job_id}.mp4")

    with open(input_path, "wb") as f:
        f.write(data)

    try:
        process_video(input_path, raw_output_path, mode=mode, duration=CLIP_SECONDS)
        _transcode_to_h264(raw_output_path, output_path)
    except Exception as exc:
        _cleanup(input_path, raw_output_path, output_path)
        raise HTTPException(500, f"Processing failed: {exc}") from exc

    _cleanup(input_path, raw_output_path)
    background_tasks.add_task(_cleanup, output_path)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="lane-detection-result.mp4",
        background=background_tasks,
    )
