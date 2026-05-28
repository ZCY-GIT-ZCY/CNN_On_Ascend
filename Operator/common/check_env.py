"""
环境检查工具 - 验证昇腾开发环境是否就绪

使用方法:
    python check_env.py
"""
import os
import sys
import subprocess
from pathlib import Path


def print_status(name, status, message=""):
    """打印检查状态"""
    symbol = "✅" if status else "❌"
    print(f"  [{symbol}] {name}")
    if message:
        print(f"      {message}")
    return status


def check_python():
    """检查 Python 版本"""
    print("\n[1] 检查 Python 环境")
    version = sys.version_info
    status = version.major == 3 and version.minor >= 8
    print_status("Python 版本", status, f"{version.major}.{version.minor}.{version.micro} (需要 >= 3.8)")
    return status


def check_cann():
    """检查 CANN 安装"""
    print("\n[2] 检查 CANN 工具链")
    cann_home = os.environ.get("ASCEND_TOOLKIT_HOME", "")
    status = bool(cann_home and Path(cann_home).exists())
    print_status("ASCEND_TOOLKIT_HOME", status, cann_home or "未设置")

    if status:
        cann_home = Path(cann_home)
        checks = [
            ("ATC 编译器", cann_home / "atc" / "bin" / "atc"),
            ("运行时库", cann_home / "lib64"),
            ("头文件", cann_home / "include"),
            ("算子库", cann_home / "opp"),
            ("Python API", cann_home / "python"),
        ]
        for name, path in checks:
            print_status(name, path.exists(), str(path))

    return status


def check_pytorch():
    """检查 PyTorch NPU"""
    print("\n[3] 检查 PyTorch NPU")
    try:
        import torch

        try:
            import torch_npu
            torch_npu.npu.init()
            npu_available = torch.npu.is_available()
        except:
            npu_available = False

        print_status("PyTorch", True, f"版本 {torch.__version__}")
        print_status("torch-npu", npu_available, f"版本 {torch_npu.__version__ if 'torch_npu' in dir() else 'N/A'}")
        print_status("NPU 可用", npu_available)

        if npu_available:
            device_count = torch.npu.device_count()
            device_name = torch.npu.get_device_name(0) if device_count > 0 else "N/A"
            print_status("NPU 设备数", True, f"{device_count}")
            print_status("NPU 设备名", True, device_name)

        return True
    except ImportError:
        print_status("PyTorch", False, "未安装")
        return False
    except Exception as e:
        print_status("PyTorch", False, str(e))
        return False


def check_mindspore():
    """检查 MindSpore"""
    print("\n[4] 检查 MindSpore")
    try:
        import mindspore as ms
        print_status("MindSpore", True, f"版本 {ms.__version__}")

        try:
            ms.context.set_context(device_target="Ascend")
            print_status("MindSpore NPU", True, "已配置")
        except:
            print_status("MindSpore NPU", False, "配置失败")

        return True
    except ImportError:
        print_status("MindSpore", False, "未安装")
        return False
    except Exception as e:
        print_status("MindSpore", False, str(e))
        return False


def check_compiler():
    """检查编译器"""
    print("\n[5] 检查编译器工具链")
    compilers = [
        ("GCC", ["gcc", "--version"]),
        ("G++", ["g++", "--version"]),
        ("CMake", ["cmake", "--version"]),
    ]

    all_ok = True
    for name, cmd in compilers:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            version_line = result.stdout.split('\n')[0] if result.stdout else "未找到"
            print_status(name, True, version_line)
        except FileNotFoundError:
            print_status(name, False, "未安装")
            all_ok = False
        except Exception as e:
            print_status(name, False, str(e))
            all_ok = False

    return all_ok


def check_npu_device():
    """检查 NPU 设备节点"""
    print("\n[6] 检查 NPU 设备")

    dev_paths = [
        "/dev/davinci0",
        "/dev/davinci_manager",
        "/dev/ascend_manager",
    ]

    all_ok = True
    for path in dev_paths:
        exists = Path(path).exists()
        status = "存在" if exists else "不存在"
        print_status(Path(path).name, exists, f"{path} - {status}")
        if not exists:
            all_ok = False

    return all_ok


def check_library_path():
    """检查库路径"""
    print("\n[7] 检查动态库路径")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    valid_paths = [p for p in ld_path.split(':') if p]
    print_status("LD_LIBRARY_PATH", bool(valid_paths),
                 f"包含 {len(valid_paths)} 个路径")

    lib_checks = [
        ("libascend_cl.so", "AscendCL"),
        ("libdrvdsmi.so", "驱动"),
    ]

    for lib_name, desc in lib_checks:
        found = False
        for path in valid_paths:
            if Path(path).exists():
                try:
                    if lib_name in os.listdir(path):
                        found = True
                        break
                except:
                    pass
        print_status(f"{lib_name} ({desc})", found)

    return True


def main():
    print("=" * 60)
    print("  昇腾算子开发环境检查工具")
    print("=" * 60)

    results = []
    results.append(("Python", check_python()))
    results.append(("CANN", check_cann()))
    results.append(("PyTorch NPU", check_pytorch()))
    results.append(("MindSpore", check_mindspore()))
    results.append(("编译器", check_compiler()))
    results.append(("NPU 设备", check_npu_device()))
    results.append(("库路径", check_library_path()))

    print("\n" + "=" * 60)
    print("  检查结果汇总")
    print("=" * 60)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        symbol = "✅" if ok else "❌"
        print(f"  {symbol} {name}")

    print(f"\n通过: {passed}/{total}")

    if passed == total:
        print("\n🎉 所有检查通过！环境已就绪，可以开始算子开发。")
        return 0
    else:
        print("\n⚠️  部分检查未通过，请修复后再继续。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
