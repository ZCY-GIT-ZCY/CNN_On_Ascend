/**
 * ACLNN 自定义算子实现 - 加法算子
 */

#include "custom_add.h"
#include "aclnn/opdev/op_log.h"
#include "aclnn/opdev/data_type_utils.h"
#include <cstring>

namespace aclnn_op {

aclnnStatus CustomAddOp::Init(const opdev::OpArgs& inputs,
                                const opdev::OpArgs& outputs,
                                const opdev::OpArgs& attrs) {
    // 获取输入张量
    auto& input_x_desc = inputs.GetArg(0);
    auto& input_y_desc = inputs.GetArg(1);
    auto& output_desc = outputs.GetArg(0);
    
    // 获取张量数据指针
    input_x_ = input_x_desc.GetData();
    input_y_ = input_y_desc.GetData();
    output_ = output_desc.GetData();
    
    // 获取形状
    shape_ = input_x_desc.GetShape();
    
    // 获取数据类型
    data_type_ = input_x_desc.GetDataType();
    
    // 获取属性 alpha
    if (attrs.Size() > 0) {
        alpha_ = attrs.GetArg(0).GetScalar<float>();
    }
    
    // 计算元素数量
    num_elements_ = 1;
    for (auto dim : shape_) {
        num_elements_ *= dim;
    }
    
    ACLNN_LOGI("CustomAddOp Init: shape=" << ShapeToString(shape_)
             << ", dtype=" << opdev::DataTypeToString(data_type_)
             << ", alpha=" << alpha_);
    
    return aclnnStatus::ACNN_SUCCESS;
}

aclnnStatus CustomAddOp::Execute(aclrtStream stream) {
    ACLNN_LOGI("CustomAddOp Execute started");
    
    switch (data_type_) {
        case opdev::DataType::FLOAT32: {
            auto* x = static_cast<float*>(input_x_);
            auto* y = static_cast<float*>(input_y_);
            auto* out = static_cast<float*>(output_);
            
            // 使用向量化操作 (伪代码，实际需要使用 NPU intrinsic)
            for (int64_t i = 0; i < num_elements_; ++i) {
                out[i] = x[i] + alpha_ * y[i];
            }
            break;
        }
        case opdev::DataType::FLOAT16: {
            // 半精度实现
            // 注意：实际生产代码应使用 NPU 向量化指令
            auto* x = static_cast<uint16_t*>(input_x_);
            auto* y = static_cast<uint16_t*>(input_y_);
            auto* out = static_cast<uint16_t*>(output_);
            
            for (int64_t i = 0; i < num_elements_; ++i) {
                // 半精度加法实现
                float fx = HalfToFloat(x[i]);
                float fy = HalfToFloat(y[i]);
                out[i] = FloatToHalf(fx + alpha_ * fy);
            }
            break;
        }
        case opdev::DataType::INT32: {
            auto* x = static_cast<int32_t*>(input_x_);
            auto* y = static_cast<int32_t*>(input_y_);
            auto* out = static_cast<int32_t*>(output_);
            
            for (int64_t i = 0; i < num_elements_; ++i) {
                out[i] = x[i] + static_cast<int32_t>(alpha_) * y[i];
            }
            break;
        }
        default:
            ACLNN_LOGE("Unsupported data type: " << opdev::DataTypeToString(data_type_));
            return aclnnStatus::ACNN_STATUS_INVALID_PARAM;
    }
    
    ACLNN_LOGI("CustomAddOp Execute completed");
    return aclnnStatus::ACNN_SUCCESS;
}

// 注册算子工厂
OP_REGISTER("CustomAdd", CustomAddOp);

} // namespace aclnn_op
