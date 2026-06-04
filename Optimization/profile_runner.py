"""用于 msprof 调用的轻量推理脚本。"""
import argparse
import time
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "common" / "acllite_utils")))
from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    resource = AclLiteResource()
    resource.init()
    model = AclLiteModel(args.model_path)
    input_data = np.zeros((224, 224, 3), dtype=np.uint8)

    for _ in range(args.warmup):
        model.execute([input_data])

    for _ in range(args.iterations):
        model.execute([input_data])
        time.sleep(0.002)

    del model
    del resource


if __name__ == "__main__":
    main()
