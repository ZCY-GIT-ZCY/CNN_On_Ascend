# ACLNN 算子开发指南

ACNN (Ascend Cloud Neural Network) 是 CANN 8.0 推荐的算子开发接口，提供了更现代化、更易用的算子开发方式。

## 环境要求

```bash
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# CANN 版本要求: 8.0.0+
atc --version
```

## 目录结构

```
03_aclnn/
├── custom_add.h        # 算子头文件
├── custom_add.cpp      # 算子实现
├── CMakeLists.txt      # CMake 构建配置
└── README.md
```

## 核心概念

### 1. OpExecutor 基类

所有 ACLNN 算子都需要继承 `opdev::OpExecutor`：

```cpp
class MyCustomOp : public opdev::OpExecutor {
public:
    const char* GetOpName() const override;
    aclnnStatus Init(const opdev::OpArgs& inputs,
                     const opdev::OpArgs& outputs,
                     const opdev::OpArgs& attrs) override;
    aclnnStatus Execute(aclrtStream stream) override;
};
```

### 2. 核心类型

| 类型 | 说明 |
|------|------|
| `AI_CORE` | 在 NPU 上执行 |
| `AI_CPU` | 在 CPU 上执行 |
| `DVPP` | 在 DVPP 硬件上执行 |

## 编译和构建

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
make install
```

## 参考资料

- [昇腾 ACLNN 开发文档](https://www.hiascend.com/document/)
