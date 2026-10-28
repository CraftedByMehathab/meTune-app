"""Export AST (AudioSet, 527 sound classes) to ONNX for the noise detector (E25), PyTorch-free at runtime.

The graph takes a raw 16 kHz mono waveform (1, samples), ≤ 10.24 s, and returns sigmoid AudioSet probabilities.
The feature extractor's Kaldi-style log-mel (numpy path of ASTFeatureExtractor: 25 ms Hann frames, 10 ms hop,
DC removal, pre-emphasis 0.97, 512-point power spectrum, 128 mels, zero-pad to 1024 frames, AudioSet mean/std)
is built into the graph, the DFT as two matmuls. Parity is checked against transformers on 10 s windows of
val_big calls. Also writes a dynamic-int8 copy (transformer MatMuls only, as for the SER models).

    uv run --group ser python eval/export_ast_onnx.py
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from metune_pipeline.decode import decode
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import ASTFeatureExtractor, ASTForAudioClassification

AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
OUT = Path(".data/models")
WIN, HOP, NFFT, FRAMES = 400, 160, 512, 1024


class WaveToProbs(torch.nn.Module):
    def __init__(self, model, fe: ASTFeatureExtractor):
        super().__init__()
        self.model = model
        n, k = np.arange(WIN)[:, None], np.arange(NFFT // 2 + 1)[None, :]
        ang = 2 * np.pi * n * k / NFFT
        self.register_buffer("cos", torch.tensor(np.cos(ang), dtype=torch.float32))
        self.register_buffer("sin", torch.tensor(-np.sin(ang), dtype=torch.float32))
        self.register_buffer("window", torch.tensor(fe.window, dtype=torch.float32))
        self.register_buffer("mel", torch.tensor(fe.mel_filters, dtype=torch.float32))      # (257, 128)
        self.mean, self.std = fe.mean, fe.std

    def forward(self, wav):
        x = wav[0]
        n = (x.shape[0] - WIN) // HOP + 1
        idx = torch.arange(n)[:, None] * HOP + torch.arange(WIN)[None, :]
        f = x[idx]
        f = f - f.mean(1, keepdim=True)
        f = torch.cat([f[:, :1] * (1 - 0.97), f[:, 1:] - 0.97 * f[:, :-1]], 1) * self.window
        power = (f @ self.cos) ** 2 + (f @ self.sin) ** 2
        fb = torch.log(torch.clamp(power @ self.mel, min=1.192092955078125e-07))
        fb = torch.nn.functional.pad(fb, (0, 0, 0, FRAMES - n))
        fb = (fb - self.mean) / (self.std * 2)
        return torch.sigmoid(self.model(input_values=fb[None]).logits)


def windows(n: int = 40) -> list[np.ndarray]:
    out = []
    for f in sorted(Path(".data/proxy/val_big").glob("*.ogg"))[:20]:
        x = decode(f, 16000)
        out += [x[i:i + int(10.24 * 16000)] for i in range(0, len(x) - 16000, 160000)]
    return out[:n]


def main() -> None:
    fe = ASTFeatureExtractor.from_pretrained(AST_ID)
    model = ASTForAudioClassification.from_pretrained(AST_ID).eval()
    wins = windows()
    with torch.inference_mode():
        ref = np.stack([torch.sigmoid(model(**fe(w, sampling_rate=16000, return_tensors="pt")).logits)[0].numpy()
                        for w in wins])
        mine = np.stack([WaveToProbs(model, fe)(torch.from_numpy(w[None]))[0].numpy() for w in wins])
    print(f"torch graph vs transformers: max |Δp| {np.abs(mine - ref).max():.5f}")

    OUT.mkdir(parents=True, exist_ok=True)
    fp32, int8 = OUT / "ast_audioset.onnx", OUT / "ast_audioset.int8.onnx"
    torch.onnx.export(WaveToProbs(model, fe), (torch.zeros(1, 160000),), str(fp32), input_names=["wav"],
                      output_names=["probs"], dynamic_axes={"wav": {1: "samples"}}, opset_version=17, dynamo=False)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                     nodes_to_exclude=_feature_matmuls(fp32))
    for path in (fp32, int8):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = np.stack([sess.run(None, {"wav": w[None].astype(np.float32)})[0][0] for w in wins])
        top = np.mean([set(np.argsort(-g)[:5]) == set(np.argsort(-r)[:5]) for g, r in zip(got, ref)])
        print(f"{path.name}: {path.stat().st_size / 1e6:.0f} MB · {len(wins)} windows · max |Δp| "
              f"{np.abs(got - ref).max():.4f} · same top-5 classes {top:.0%}")


def _feature_matmuls(path: Path) -> list[str]:
    """The DFT / mel matmuls stay fp32: quantizing them distorts the spectrogram itself."""
    import onnx
    g = onnx.load(str(path)).graph
    feats = {"cos", "sin", "mel"}
    return [n.name for n in g.node if n.op_type == "MatMul" and any(i.split(".")[-1] in feats or i in feats
                                                                     for i in n.input)]


if __name__ == "__main__":
    main()
