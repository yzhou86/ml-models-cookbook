# Interactive model sites

Each directory is a standalone static site. The deployable entry point is
`dist/index.html`; `.openai/hosting.json` records the hosting configuration
for sites that use Sites deployment.

| Site | Model focus | Main interaction |
| --- | --- | --- |
| `yolo11n-demo` | YOLO11n detection | Confidence, IoU, input size, and NMS effects |
| `whisper-tutorial` | Whisper transcription | Model selection, tuning trade-offs, and WER calculation |
| `silero-vad-tutorial` | Silero VAD tutorial | Guided explanation of VAD concepts |
| `mobilenet-v3-lab` | MobileNetV3 classification | Variant, input size, classification threshold, and architecture stages |
| `ecapa-speaker-lab` | ECAPA-TDNN verification | Query condition, utterance length, similarity threshold, and FAR/FRR trade-off |
| `arcface-face-lab` | SCRFD + ArcFace verification | Detection gate, alignment condition, cosine score, and acceptance threshold |
| `silero-vad-lab` | Silero VAD streaming | Frame probability, hysteresis threshold, minimum speech, and hangover |

The newer labs use repository sample assets and front-end simulations to make
model behavior inspectable without downloading or running model weights in the
browser. Any decision thresholds shown in the sites are teaching controls, not
production calibration values.
