env  
python 3.10  
conda activate rinna-env - 切换到训练环境  
nvidia-smi -l 1 - 查看显卡配置  
\\u.{4} - [URL](https://blog.cloudnative.co.jp/23733/)  

## 项目介绍
本项目，为日语N1单词学习辅助生成式AI，目的是通过特殊训练后的文生文AI生成N1的现代小说故事来辅助用户学习和记忆N1相关的知识点。

## Model
[HuggingFace repo](https://huggingface.co/BBusagi/sft-model-n1)

##
### TODO
我先用了复杂的语料（[train2](output/train2/train_data_with_prompt.jsonl)）训练获得了[ckpoint2880](sft-model-n1/medium/checkpoint-2880),表现良好。
但是反过来继续用简单语料[train1](output/train1/train_data_with_prompt.jsonl)训练之后获得的[ckpoint1550](sft-model-n1/medium/checkpoint-1500)表现就很差。
为什么低质量语料会对已经过训练的模型有这么大的负面影响，理论上不是应该继续提升，但提升很小吗？

### Manual
+ Step00 环境配置和检测
    + [Tool_check_api_connection](Script/Tool_check_api_connection.py)
+ Step01 准备语料
    + [Step1_generate_json](Script/Step1_generate_json.py)
    + [Tool_convert_jsonl_to_txt](Script/Tool_convert_jsonl_to_txt.mjs)
    + [Step1_prepare_training_data](Script/Step1_prepare_training_data.py)
+ Step02 训练模型 断点保存
    + [Tool_check_gpu_status](Script/Tool_check_gpu_status.py)
    + [Step2_train_sft_model](Script/Step2_train_sft_model.py)
    + [Step2_save_model](Script/Step2_save_model.py)
+ Step03 模型评估
    + [Step3_infer_and_evaluate](Script/Step3_infer_and_evaluate.py)
    + [Tool_run_model_inference](Script/Tool_run_model_inference.py)
    + [Step3_compare_multiple_models](Script/Step3_compare_multiple_models.py)

### DevLog
    + 模型状态： 未经训练 很离谱 无法使用
+ ✅Step01  chatGPT+rinnasmall
    + 环境配置
    + OpenAPI测试和rinna测试
    + 尝试Batch调用
    + Batch检测，生成1000个故事
    + 修改prompt，到达满意阶段
    + 预处理语料，生成训练JSON
    + sft训练尝试
    + 测试：**可行 模仿语料成功 长度控制有问题 整体文章较肤浅**

+ ✅Step02  gpt4长文语料 + rinnamedium
    + prompt修改更新
    + batch更新 
    >BUG Limit: 90,000 enqueued tokens - 4o更换为4omini
    + 预处理语料，生成训练JSON
    + sft训练尝试
    > BUG 全参调节 长度不一致 - 参数提纯
    + 全参调节（train1）
    > BUG 系统蓝屏 显存溢出 - 重新pythor调用GPU
    + 断点保存 多线程开启
    + 每次运行 90 分钟训练 
    + 测试：checkpoint-2880 -> **表现状态优秀，生成内容符合标准**
    + 断点续练
    > BUG lm_head
    > 1. 缺失模型参数权重文件（pytorch—_model.bin）
    > -- 修改save模块 重新小片段训练 (尝试LORA) 
    > -- 仍然存在
    > 2. resume_from_checkpoint = 老版本，默认只支持.bin；新版本保存格式为.safetensors
    > -- trainer.train(~~resume_from_checkpoint~~)
    > -- ✅已解决
    + sft训练
    + 检测：checkpoint-1500 -> **模型状态回滚，生成长度不足，效果极差**

### Roadmap
+ 训练监控 完整性检测相关模组
+ 检查剩余步骤，输出“预计还需 X 次训练” 
+ 记录每次的 step、耗时、loss 到日志文件
+ API部署
+ Fultter部署
+ 加入评级系统
> “浓度评分”更适合在 生成完成后做自动评分或筛选，而不是一开始就塞进训练目标中
+ 加入文字加粗等
+ 情节控制机制
+ 后期1 若训练资源允许，可尝试知识蒸馏 + 链式优化，打造轻量高效模型
+ 后期2 构建自己的“日语学习大模型”，支持输入 JLPT 级别 + 输出个性化文本

This is a test to dev branch