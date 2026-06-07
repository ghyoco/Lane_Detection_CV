---
title: Lane Detection API
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

FastAPI backend for the COMP7116001 Computer Vision lane-detection project.

`POST /api/process` — multipart form with a `video` file field and optional
`mode` field (`auto` | `day` | `night`, default `auto`). Returns the same clip
(first ~10–30 seconds) with a Bird's-Eye-View lane overlay burned in as H.264 mp4.
