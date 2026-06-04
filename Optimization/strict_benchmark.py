"""
严格控制的独立进程基准测试。

设计原则：
1. 每个 OM 推理在一个全新进程中运行
2. 每一轮测量都使用独立 Python 子进程，避免 ACL 重复 init/finalize 污染
3. 使用固定 seed 的伪随机 uint8 输入，保证可复现但非全零
4. 预热与正式测量在同一子进程内完成，确保稳定态测量
5. 支持多轮重复采样并输出各轮统计，便于判断结果稳定性
6. 父进程只负责调度与汇总，不直接持有 ACL 资源
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
FIXED_SEED = 42


def make_deterministic_input(height=224, width=224, channels=3, seed=FIXED_SEED):
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)


def run_child_once(model_path: Path, warmup_first: int, warmup_full: int, iterations: int) -> dict:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str((_Path(__file__).resolve().parent.parent / "common" / "acllite_utils")))
    from acllite_resource import AclLiteResource
    from acllite_model import AclLiteModel

    resource = AclLiteResource()
    resource.init()
    model = AclLiteModel(str(model_path))
    input_data = make_deterministic_input()

    for _ in range(warmup_first):
        model.execute([input_data])
    for _ in range(warmup_full):
        model.execute([input_data])

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.execute([input_data])
        times.append((time.perf_counter() - start) * 1000)

    del model
    del resource

    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "std_ms": float(times.std()),
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "fps": float(1000.0 / times.mean()),
        "sample_count": len(times),
    }


def run_round_in_subprocess(model_path: Path, warmup_first: int, warmup_full: int, iterations: int) -> dict:
    cmd = [
        sys.executable,
        str(THIS_FILE),
        "--child-once",
        "--model-path",
        str(model_path),
        "--warmup-first",
        str(warmup_first),
        "--warmup-full",
        str(warmup_full),
        "--iterations",
        str(iterations),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip())

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"No JSON output found: {proc.stdout[-2000:]}")


def benchmark(model_path: Path, warmup_first: int, warmup_full: int,
              iterations: int, rounds: int,
              cooldown_between_rounds: float) -> dict:
    per_round = []
    for r in range(rounds):
        result = run_round_in_subprocess(
            model_path=model_path,
            warmup_first=warmup_first,
            warmup_full=warmup_full,
            iterations=iterations,
        )
        per_round.append(result)
        if r < rounds - 1 and cooldown_between_rounds > 0:
            time.sleep(cooldown_between_rounds)

    means = [r["mean_ms"] for r in per_round]
    return {
        "rounds": rounds,
        "iterations_per_round": iterations,
        "warmup_first": warmup_first,
        "warmup_full": warmup_full,
        "overall_mean_ms": float(np.mean(means)),
        "overall_std_ms": float(np.std(means)),
        "round_means": [float(m) for m in means],
        "per_round": per_round,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="严格控制的独立进程基准测试")
    parser.add_argument("--model-key", default="", help="模型标识，如 mobilenet_v3_small")
    parser.add_argument("--model-path", required=True, help="OM 文件路径")
    parser.add_argument("--warmup-first", type=int, default=30)
    parser.add_argument("--warmup-full", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cooldown-between-rounds", type=float, default=2.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--child-once", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path)

    if args.child_once:
        result = run_child_once(
            model_path=model_path,
            warmup_first=args.warmup_first,
            warmup_full=args.warmup_full,
            iterations=args.iterations,
        )
        print(json.dumps(result, ensure_ascii=False))
        return

    result = benchmark(
        model_path=model_path,
        warmup_first=args.warmup_first,
        warmup_full=args.warmup_full,
        iterations=args.iterations,
        rounds=args.rounds,
        cooldown_between_rounds=args.cooldown_between_rounds,
    )
    result["model_key"] = args.model_key
    result["model_path"] = str(model_path)

    out_str = json.dumps(result, ensure_ascii=False)
    print(out_str, file=sys.stderr)
    print(out_str)

    if args.output:
        Path(args.output).write_text(out_str, encoding="utf-8")


if __name__ == "__main__":
    main()
