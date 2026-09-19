# FFmpeg 极简集（构建资产，随仓库分发）

用途：PyInstaller 打包时**替换** decord wheel 自带的 FFmpeg 全家桶
（184MB → 14MB，−92%），使发布产物过 400MB 体积门禁。

## 来源与配方

按 `DEPENDENCIES.md`「极简 FFmpeg 构建配方」（MSYS2 ucrt64，ffmpeg-9.0
官方源码）重建：

```
./configure --enable-shared --disable-static --disable-programs --disable-doc \
  --disable-debug --disable-network --disable-autodetect --disable-avdevice \
  --disable-everything --enable-swscale --enable-swresample --enable-libdav1d \
  --enable-decoder=h264,hevc,vp9,libdav1d \
  --enable-parser=h264,hevc,av1,vp9 \
  --enable-demuxer=mov,matroska,avi --enable-protocol=file \
  --enable-filter=setparams,crop,scale,transpose,format,null \
  --extra-ldflags="-static-libgcc"
```

必须含 decord 所需 bsf：null / h264_mp4toannexb / hevc_mp4toannexb /
mpeg4_unpack_bframes / vp9_superframe_split（缺则 NVDEC 路径加载失败）。
**AV1 必须用 libdav1d**（native av1 在此构建下 send_packet -40 且病态慢）。

## 等价性证据（DEPENDENCIES.md 已实测）

- CPU 软解：h264 99.6% / hevc 98.7% / av1 98.4%（av1 系 dav1d vs native 换代）
- NVDEC：三码 99.9~100.1% **零差异**
- hybrid 混跑：两轮 6+6 交错全面交叉 = 平价
- E2E 门禁：h264 真值 7223/7223、hevc/av1 全片 hybrid 通过

## 使用

`RaceVideoToLog.spec` 自动检测：环境变量 `RACELOG_FFMPEG_MIN` 优先，
否则用本目录（`assets/ffmpeg_min/`）；目录缺失时回退 decord 全量集并打印
提示（不阻断构建，只是产物大 ~170MB）。

License = LGPL（与 GPL 全家桶不同，见 DEPENDENCIES.md）。
