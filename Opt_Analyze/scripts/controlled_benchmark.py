"""
受控实验基准测试框架 - Opt_Analyze 主测量脚本

核心改进：
1. ✅ 环境监控（温度、负载、NPU状态）伴随每次测量
2. ✅ 严格子进程隔离（每个变体独立进程）
3. ✅ 随机化顺序消除顺序偏差
4. ✅ 轮间冷却（cooldown）
5. ✅ 记录环境状态作为测量元数据
6. ✅ 与原始 baseline 和 reexport_plain 的正确对比

用法：
  # 完整实验
  python scripts/controlled_benchmark.py --mode full

  # 快速验证（仅 baseline vs fp16 vs reexport_plain）
  python scripts/controlled_benchmark.py --mode quick

  # 仅环境监控测试
  python scripts/controlled_benchmark.py --mode monitor
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 确保可以找到项目模块
SCRIPT_DIR = Path(__file__).resolve().parent
OPT_ANALYZE = SCRIPT_DIR.parent
CNN_ROOT = OPT_ANALYZE.parent
COMMON_DIR = CNN_ROOT / "common" / "acllite_utils"
OPTIMIZATION_DIR = CNN_ROOT / "Optimization"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(OPTIMIZATION_DIR))

try:
    from config import BASELINES, MODELS_DIR
except ImportError:
    # 回退定义
    BASELINES = {}
    MODELS_DIR = OPTIMIZATION_DIR / "models"

from env_monitor import EnvMonitor


# ---------- 路径常量 ----------
MOBILENET_DIR = CNN_ROOT / "MobileNet"
RESNET_DIR = CNN_ROOT / "ResNet"
RESULTS_DIR = OPT_ANALYZE / "data"
LOGS_DIR = OPT_ANALYZE / "logs"
REPORTS_DIR = OPT_ANALYZE / "reports"

for d in [RESULTS_DIR, LOGS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 基准模型路径
BASELINE_OMS = {
    "mobilenet_v3_small": MOBILENET_DIR / "mobilenetv3_aipp.om",
    "resnet50": RESNET_DIR / "resnet50_aipp.om",
}

ATC_BIN = Path(os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest")) / "atc" / "bin" / "atc"
FIXED_SEED = 42


# ---------- 数据结构 ----------
@dataclass
class BenchmarkSample:
    """单次 benchmark 测量结果。"""
    model_key: str
    variant_name: str
    round_idx: int
    iteration_in_round: int
    latency_ms: float
    env_before: Dict = field(default_factory=dict)
    env_after: Dict = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class RoundResult:
    """一轮完整 benchmark 结果（多次 iteration）。"""
    model_key: str
    variant_name: str
    round_idx: int
    warmup_iterations: int
    measure_iterations: int
    latencies_ms: List[float] = field(default_factory=list)
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    fps: float = 0.0
    env_summary: Dict = field(default_factory=dict)
    error: str = ""
    timestamp: str = ""


# ---------- 模型执行 ----------
def run_inference_child(model_path: Path, warmup: int, iterations: int,
                        measure_callback: bool = False) -> Dict:
    """
    在子进程中运行推理，返回测量结果。
    这个函数被设计为通过 subprocess 调用，以隔离 ACL 状态。
    """
    sys.path.insert(0, str(COMMON_DIR))
    import numpy as np
    from acllite_resource import AclLiteResource
    from acllite_model import AclLiteModel

    resource = AclLiteResource()
    resource.init()
    model = AclLiteModel(str(model_path))

    # 固定随机输入（可复现但非全零）
    rng = np.random.Generator(np.random.PCG64(FIXED_SEED))
    input_data = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    # 预热
    for _ in range(warmup):
        model.execute([input_data])

    # 正式测量
    latencies = []
    for i in range(iterations):
        start = time.perf_counter()
        model.execute([input_data])
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    del model
    del resource

    lat_arr = np.array(latencies)
    result = {
        "latencies_ms": [float(x) for x in latencies],
        "mean_ms": float(lat_arr.mean()),
        "min_ms": float(lat_arr.min()),
        "max_ms": float(lat_arr.max()),
        "std_ms": float(lat_arr.std()),
        "p50_ms": float(np.percentile(lat_arr, 50)),
        "p90_ms": float(np.percentile(lat_arr, 90)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "p99_ms": float(np.percentile(lat_arr, 99)),
        "fps": float(1000.0 / lat_arr.mean()),
        "sample_count": len(latencies),
    }
    return result


def run_one_round(model_path: Path, warmup: int, iterations: int) -> Dict:
    """在独立子进程中执行一轮 benchmark。"""
    # 用一个哨兵环境变量来触发子进程模式，避免参数解析问题
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "controlled_benchmark.py"),
        "--mode", "child",
        "--model-path", str(model_path),
        "--warmup", str(warmup),
        "--iterations", str(iterations),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"Child process failed: {proc.stderr[:2000]}")

    # 从 stdout 提取 JSON
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"No JSON result: {proc.stdout[:2000]}")


# ---------- ATC 编译 ----------
def compile_to_om(model_key: str, onnx_path: Path, output_prefix: Path,
                  aipp_cfg: Optional[Path] = None) -> Dict:
    """使用 ATC 编译 ONNX 到 OM。"""
    if aipp_cfg is None:
        model_dir = MOBILENET_DIR if "mobilenet" in model_key else RESNET_DIR
        aipp_cfg = model_dir / "aipp.cfg"

    cmd = [
        str(ATC_BIN),
        f"--model={onnx_path}",
        "--framework=5",
        f"--output={output_prefix}",
        "--soc_version=Ascend310B1",
        "--input_format=NCHW",
        "--input_shape=input_image:1,3,224,224",
        "--precision_mode=allow_mix_precision",
        "--op_select_implmode=high_performance",
        "--buffer_optimize=l2_optimize",
        "--enable_small_channel=1",
        f"--insert_op_conf={aipp_cfg}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    om_path = output_prefix.with_suffix(".om")

    # 收集编译日志中的警告/异常
    warnings = []
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if any(kw in line.lower() for kw in ["warning", "error", "fail", "fallback"]):
            warnings.append(line.strip())

    return {
        "status": "ok" if proc.returncode == 0 and om_path.exists() else "failed",
        "om_path": str(om_path),
        "stdout": proc.stdout[-3000:],
        "stderr": proc.stderr[-3000:],
        "warnings": warnings[:50],
        "returncode": proc.returncode,
    }


# ---------- 实验定义 ----------
def get_variants() -> List[Dict]:
    """返回所有要测试的变体定义。"""
    return [
        {"id": "baseline_copy", "name": "原始 AIPP OM 基线", "type": "om"},
        {"id": "reexport_plain", "name": "重新导出版（纯 ONNX 副本重编译）", "type": "onnx"},
        {"id": "fp16", "name": "FP16 精度", "type": "onnx"},
        {"id": "shape_inferred", "name": "Shape 推断", "type": "onnx"},
        {"id": "opset13", "name": "Opset 13", "type": "onnx"},
        {"id": "bn_folded", "name": "BN 折叠", "type": "onnx"},
        {"id": "channels_last", "name": "Channels Last 布局", "type": "onnx"},
        {"id": "onnxsim", "name": "ONNX 简化", "type": "onnx"},
        {"id": "onnxoptimizer", "name": "ONNX 优化器", "type": "onnx"},
    ]


def get_om_path(model_key: str, variant_id: str) -> Path:
    """获取变体对应的 OM 文件路径。"""
    if variant_id == "baseline_copy":
        return BASELINE_OMS[model_key]

    if variant_id == "reexport_plain":
        return OPTIMIZATION_DIR / "models" / f"{model_key}_reexport_plain.om"

    return OPTIMIZATION_DIR / "models" / f"{model_key}_{variant_id}.om"


def get_onnx_path(model_key: str, variant_id: str) -> Path:
    """获取变体对应的 ONNX 文件路径。"""
    if variant_id == "reexport_plain":
        # 直接复用项目中的原始 ONNX（非优化变体）
        if model_key == "mobilenet_v3_small":
            return MOBILENET_DIR / "mobilenetv3.onnx"
        else:
            return RESNET_DIR / "resnet50.onnx"

    return OPTIMIZATION_DIR / "models" / f"{model_key}_{variant_id}.onnx"


# ---------- 主 benchmark 循环 ----------
def run_controlled_benchmark(model_key: str, variant_ids: List[str],
                             warmup: int = 30, iterations: int = 180,
                             rounds: int = 3, cooldown: float = 3.0,
                             random_order: bool = True,
                             label: str = "") -> List[RoundResult]:
    """
    受控 benchmark 核心函数。

    对每个变体运行 rounds 轮，每轮在独立子进程中执行。
    支持随机化顺序、轮间冷却、环境监控。
    """
    logger = EnvMonitor(log_dir=str(LOGS_DIR), label=label)

    # 确定运行顺序
    run_order = list(variant_ids)
    if random_order:
        random.shuffle(run_order)

    all_results: List[RoundResult] = []

    for variant_id in run_order:
        om_path = get_om_path(model_key, variant_id)

        if not om_path.exists():
            all_results.append(RoundResult(
                model_key=model_key, variant_name=variant_id,
                round_idx=0, warmup_iterations=warmup,
                measure_iterations=iterations,
                error=f"OM not found: {om_path}",
                timestamp=datetime.now().isoformat(),
            ))
            continue

        print(f"\n{'='*60}")
        print(f"[{model_key}] 变体: {variant_id}  ({om_path})")
        print(f"{'='*60}")

        for r in range(rounds):
            logger.start(interval=2.0)
            time.sleep(0.5)  # 让采集器稳定

            round_result = RoundResult(
                model_key=model_key,
                variant_name=variant_id,
                round_idx=r,
                warmup_iterations=warmup,
                measure_iterations=iterations,
                timestamp=datetime.now().isoformat(),
            )

            # 记录环境前置状态
            round_result.env_summary["before"] = logger.get_summary_stats()

            try:
                stats = run_one_round(
                    model_path=om_path,
                    warmup=warmup,
                    iterations=iterations,
                )
                # 填充结果
                for key in ["mean_ms", "min_ms", "max_ms", "std_ms",
                            "p50_ms", "p90_ms", "p95_ms", "p99_ms", "fps"]:
                    setattr(round_result, key, stats.get(key, 0.0))
                round_result.latencies_ms = stats.get("latencies_ms", [])
                print(f"  Round {r+1}/{rounds}: {round_result.mean_ms:.4f} ms "
                      f"(min={round_result.min_ms:.4f}, max={round_result.max_ms:.4f}, "
                      f"std={round_result.std_ms:.4f})")

            except Exception as e:
                round_result.error = str(e)
                print(f"  Round {r+1}/{rounds}: FAILED - {e}")

            # 记录环境后置状态
            logger.stop()
            round_result.env_summary["after"] = logger.get_summary_stats()

            all_results.append(round_result)

            # 轮间冷却
            if r < rounds - 1 and cooldown > 0:
                print(f"  Cooling {cooldown}s...")
                time.sleep(cooldown)

    return all_results


# ========== 主入口 ==========
def main():
    parser = argparse.ArgumentParser(description="Opt_Analyze 受控基准测试框架")
    parser.add_argument("--mode", default="quick",
                        choices=["quick", "full", "monitor", "child"])
    # 子进程参数
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument("--random-order", action="store_true", default=True)
    parser.add_argument("--no-random-order", action="store_false", dest="random_order")
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    # 子进程模式：执行一次推理并返回 JSON
    if args.mode == "child":
        if not args.model_path:
            print(json.dumps({"error": "child mode requires --model-path"}))
            return
        result = run_inference_child(
            model_path=Path(args.model_path),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(result, ensure_ascii=False))
        return

    # 监控模式：仅运行环境监控
    if args.mode == "monitor":
        print("Running environment monitor for 30s...")
        monitor = EnvMonitor(log_dir=str(LOGS_DIR), label="standalone")
        monitor.start(interval=1.0)
        time.sleep(30)
        monitor.stop()
        print(monitor.report())
        return

    # ---------- 主实验模式 ----------
    # 先检查环境基线
    print("=" * 60)
    print("Opt_Analyze 受控基准测试")
    print("=" * 60)
    print(f"时间: {datetime.now().isoformat()}")
    print(f"模式: {args.mode}")

    # 记录初始环境状态
    monitor = EnvMonitor(log_dir=str(LOGS_DIR), label="pre_check")
    monitor.start(interval=1.0)
    time.sleep(5)
    monitor.stop()
    print(monitor.report())

    models = ["mobilenet_v3_small", "resnet50"]

    if args.mode == "quick":
        # 快速模式：仅 baseline vs fp16 vs reexport_plain
        variants = ["baseline_copy", "reexport_plain", "fp16"]
        rounds = args.rounds
    elif args.mode == "full":
        # 完整模式：所有变体
        variants = ["baseline_copy", "reexport_plain", "fp16",
                    "shape_inferred", "opset13", "bn_folded",
                    "channels_last", "onnxsim", "onnxoptimizer"]
        rounds = args.rounds
    else:
        variants = ["baseline_copy"]
        rounds = 1

    all_data = {}
    for model_key in models:
        print(f"\n{'#'*60}")
        print(f"# 模型: {model_key}")
        print(f"{'#'*60}")

        results = run_controlled_benchmark(
            model_key=model_key,
            variant_ids=variants,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=rounds,
            cooldown=args.cooldown,
            random_order=args.random_order,
            label=f"{model_key}_{args.mode}",
        )
        all_data[model_key] = [asdict(r) for r in results]

    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"controlled_benchmark_{ts}.json"
    result_file.write_text(
        json.dumps(all_data, indent=2, ensure_ascii=False)
    )
    print(f"\n结果已保存: {result_file}")

    # 生成快速摘要
    print("\n" + "=" * 60)
    print("快速摘要")
    print("=" * 60)
    for model_key, results in all_data.items():
        print(f"\n--- {model_key} ---")
        by_variant = {}
        for r in results:
            by_variant.setdefault(r["variant_name"], []).append(r)

        for vid, rounds_data in by_variant.items():
            means = [r["mean_ms"] for r in rounds_data if not r["error"]]
            if means:
                print(f"  {vid:20s}: {sum(means)/len(means):.4f} ms "
                      f"(over {len(means)} rounds, "
                      f"min_round={min(means):.4f}, max_round={max(means):.4f})")
            else:
                print(f"  {vid:20s}: ERROR - {rounds_data[0]['error']}")


if __name__ == "__main__":
    main()
