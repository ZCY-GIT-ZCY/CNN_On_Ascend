"""独立进程基准测试，避免 ACL 多次初始化造成串扰。"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "common" / "acllite_utils")))
from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel


def benchmark(model_path: Path, warmup: int, iterations: int) -> dict:
    resource = AclLiteResource()
    resource.init()
    model = AclLiteModel(str(model_path))
    input_data = np.zeros((224, 224, 3), dtype=np.uint8)

    for _ in range(warmup):
        model.execute([input_data])

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.execute([input_data])
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    result = {
        "mean_ms": float(times.mean()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "std_ms": float(times.std()),
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "fps": float(1000.0 / times.mean()),
    }

    del model
    del resource
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=120)
    args = parser.parse_args()

    result = benchmark(Path(args.model_path), warmup=args.warmup, iterations=args.iterations)
    print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
