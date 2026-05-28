/**
 * ACLNN 自定义算子示例 - 加法算子
 * 
 * ACLNN 是 CANN 8.0 推荐的算子开发接口
 * 使用 C++ 实现，可同时支持 CPU 和 NPU 执行
 */

#ifndef CUSTOM_ADD_H
#define CUSTOM_ADD_H

#include <vector>
#include "aclnn/opdev/op_executor.h"
#include "aclnn/opdev/op_def.h"
#include "aclnn/opdev/common_types.h"
#include "aclnn/opdev/op_arg_def.h"

namespace aclnn_op {

/**
 * 自定义加法算子
 * 计算: output = input_x + alpha * input_y
 */
class CustomAddOp : public opdev::OpExecutor {
public:
    CustomAddOp() = default;
    ~CustomAddOp() override = default;
    
    // 获取算子名称
    const char* GetOpName() const override {
        return "CustomAdd";
    }
    
    // 初始化参数
    aclnnStatus Init(const opdev::OpArgs& inputs,
                     const opdev::OpArgs& outputs,
                     const opdev::OpArgs& attrs) override;
    
    // 前向传播
    aclnnStatus Execute(aclrtStream stream) override;
    
    // 获取核心类型 (AI_CORE = NPU, AI_CPU = CPU)
    op::CoreType GetCoreType() const override {
        return op::CoreType::AI_CORE;
    }
    
    // 获取实现模式
    op::OpImplMode GetOpImplMode() const override {
        return op::OpImplMode::IMPL_MODE_DEFAULT;
    }

private:
    // 输入输出指针
    void* input_x_ = nullptr;
    void* input_y_ = nullptr;
    void* output_ = nullptr;
    
    // 张量形状
    std::vector<int64_t> shape_;
    
    // 数据类型
    opdev::DataType data_type_;
    
    // 属性
    float alpha_ = 1.0f;
    
    // 元素数量
    int64_t num_elements_ = 1;
};

} // namespace aclnn_op

#endif // CUSTOM_ADD_H
