#!/usr/bin/env python3
"""
快速测试脚本 - 验证各种算子开发方式

使用方法:
    python3 run_tests.py [test_name]
    
参数:
    test_name: 可选，指定要运行的测试
              - pytorch: 测试 PyTorch NPU 算子
              - mindspore: 测试 MindSpore 算子
              - all: 运行所有测试 (默认)
"""
import os
import sys
import subprocess


def run_test(name, command, description):
    """运行单个测试"""
    print(f"\n{'='*60}")
    print(f"  测试: {name}")
    print(f"  说明: {description}")
    print('='*60)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=False,
            timeout=120
        )
        
        if result.returncode == 0:
            print(f"\n✅ {name} 测试通过")
            return True
        else:
            print(f"\n❌ {name} 测试失败 (退出码: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n❌ {name} 测试超时")
        return False
    except Exception as e:
        print(f"\n❌ {name} 测试出错: {e}")
        return False


def main():
    # 获取当前目录
    operator_dir = os.path.dirname(os.path.abspath(__file__))
    
    tests = {
        "pytorch": {
            "command": f"cd {operator_dir}/01_pytorch_npu && python3 custom_ops.py",
            "description": "测试 PyTorch NPU 自定义算子"
        },
        "mindspore": {
            "command": f"cd {operator_dir}/04_mindspore && python3 custom_op.py",
            "description": "测试 MindSpore 自定义算子"
        },
        "env": {
            "command": f"python3 {operator_dir}/common/check_env.py",
            "description": "运行环境检查"
        }
    }
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        if test_name == "all":
            test_list = tests
        elif test_name in tests:
            test_list = {test_name: tests[test_name]}
        else:
            print(f"未知测试: {test_name}")
            print(f"可用测试: {', '.join(tests.keys())}, all")
            return 1
    else:
        test_list = tests
    
    # 设置环境变量
    cann_script = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    if os.path.exists(cann_script):
        os.system(f"source {cann_script} > /dev/null 2>&1")
    
    # 运行测试
    print("="*60)
    print("  算子开发快速测试")
    print("="*60)
    print(f"\n将运行 {len(test_list)} 个测试...")
    
    results = []
    for name, info in test_list.items():
        result = run_test(name, info["command"], info["description"])
        results.append((name, result))
    
    # 汇总
    print("\n" + "="*60)
    print("  测试结果汇总")
    print("="*60)
    
    for name, ok in results:
        symbol = "✅" if ok else "❌"
        print(f"  {symbol} {name}")
    
    passed = sum(1 for _, ok in results if ok)
    print(f"\n通过: {passed}/{len(results)}")
    
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
