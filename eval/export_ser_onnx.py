"""Export the fine-tuned SER model (E16) to ONNX so the pipeline runs it with onnxruntime, PyTorch-free (DR-020).

The graph takes a raw 16 kHz mono waveform (1, samples) — the feature extractor has do_normalize=false, so
no preprocessing — and returns softmax probabilities in meTune TONES order. Also writes a dynamic-int8
copy (~4× smaller, faster on CPU). Parity is checked on real speech windows from the val calls: the exported
model must give the same argmax as PyTorch on every window.

    uv run --group ser python eval/export_ser_onnx.py --model .data/proxy/ser_ft/E16-ft/model
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from metune_pipeline.config import load_config
from metune_pipeline.decode import decode
from metune_pipeline.schema import TONES
from metune_pipeline.ser import speech_windows
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForAudioClassification

OUT = Path(".data/models")


class Probs(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wav):
        return self.model(input_values=wav).logits.softmax(-1)


def windows(n: int = 60) -> list[np.ndarray]:
    cfg, out = load_config(), []
    for f in sorted(Path(".data/proxy/val").glob("*.ogg")):
        x = decode(f, 16000)
        out += [x[s:e] for s, e in speech_windows(x, cfg.vad_threshold)]
        if len(out) >= n:
            break
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="ser_e16")
    a = ap.parse_args()
    model = AutoModelForAudioClassification.from_pretrained(a.model).eval()
    assert [model.config.id2label[i] for i in range(len(TONES))] == TONES, "label order must match TONES"

    # Reference BEFORE exporting: tracing leaves the module altered (first parity check compared against that).
    wins = windows()
    with torch.inference_mode():
        ref = np.stack([Probs(model)(torch.from_numpy(w[None]).float())[0].numpy() for w in wins])

    OUT.mkdir(parents=True, exist_ok=True)
    fp32, int8 = OUT / f"{a.name}.onnx", OUT / f"{a.name}.int8.onnx"
    torch.onnx.export(Probs(model), (torch.zeros(1, 32000),), str(fp32), input_names=["wav"], output_names=["probs"],
                      dynamic_axes={"wav": {1: "samples"}}, opset_version=17, dynamo=False)
    # Quantizing every op (incl. the CNN feature encoder) broke predictions (58% same argmax); transformer
    # MatMuls only is the usual wav2vec2 recipe.
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    for path in (fp32, int8):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = np.stack([sess.run(None, {"wav": w[None].astype(np.float32)})[0][0] for w in wins])
        same = float(np.mean(got.argmax(1) == ref.argmax(1)))
        print(f"{path.name}: {path.stat().st_size / 1e6:.0f} MB · {len(wins)} windows · max |Δp| "
              f"{np.abs(got - ref).max():.4f} · same argmax {same:.0%}")


if __name__ == "__main__":
    main()
