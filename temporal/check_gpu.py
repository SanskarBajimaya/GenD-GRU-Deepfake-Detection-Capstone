"""Check CUDA computation and the actual RetinaFace ONNX session."""
import sys

import numpy as np
import torch
import onnxruntime

from src.retinaface import prepare_model


def main():
    print("Python:", sys.version.split()[0], sys.executable)
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("cuDNN:", torch.backends.cudnn.version())
    print("ONNX Runtime version:", onnxruntime.__version__)
    print("Available ONNX providers:", onnxruntime.get_available_providers())

    torch_ok = False
    if torch.cuda.is_available():
        try:
            x = torch.ones((2, 2), device="cuda")
            assert (x @ x).sum().item() == 8
            torch.cuda.synchronize()
            torch_ok = True
        except Exception as exc:
            print("PyTorch CUDA computation failed:", exc)

    retina_ok = False
    try:
        detector = prepare_model(det_thres=0.1)
        provider = detector.session.get_providers()[0]
        # Exercise inference as well as DLL loading/session initialization.
        detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        retina_ok = provider == "CUDAExecutionProvider"
    except Exception as exc:
        print("RetinaFace initialization/inference failed:", exc)
        provider = "unavailable"

    print()
    print("GPU SETUP OK" if torch_ok and retina_ok else "GPU SETUP INCOMPLETE")
    print("PyTorch:", "CUDA" if torch_ok else "CPU or CUDA unavailable")
    print("RetinaFace:", provider)
    if not torch_ok:
        print("PyTorch needs a compatible CUDA build and working NVIDIA driver.")
    if not retina_ok:
        print("RetinaFace needs working CUDA 12.x/cuDNN 9.x DLLs and CUDAExecutionProvider.")
    return 0 if torch_ok and retina_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
