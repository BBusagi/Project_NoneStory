env
python 3.10
conda activate rinna-env - 切换到训练环境
nvidia-smi -l 1 - 查看显卡配置
\\u.{4} - [URL](https://blog.cloudnative.co.jp/23733/)

### Manual
+ Step00 环境配置和检测
    + [text](Script/Tool_check_api_connection.py)
+ Step01 准备语料
    + [text](Script/Step1_generate_json.py)
    + [text](Script/Tool_convert_jsonl_to_txt.mjs)
    + [text](Script/Step1_prepare_training_data.py)
+ Step02 训练模型 断点保存
    + [text](Script/Tool_check_gpu_status.py)
    + [text](Script/Step2_train_sft_model.py)
    + [text](Script/Step2_save_model.py)
+ Step03 模型评估
    + [text](Script/Step3_infer_and_evaluate.py)
    + [text](Script/Tool_run_model_inference.py)
    + [text](Script/Step3_compare_multiple_models.py)

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
    + 模型使用检测：可行 模仿语料成功 长度控制有问题 整体文章较肤浅

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
    + 验证模型
    + 断点续练
    > BUG lm_head
    > 1. 缺失模型参数权重文件（pytorch—_model.bin）
    > -- 修改save模块 重新小片段训练 (尝试LORA) 
    > -- 仍然存在
    > 2. resume_from_checkpoint = 老版本，默认只支持.bin；新版本保存格式为.safetensors
    > -- trainer.train(~~resume_from_checkpoint~~)
    > -- ✅已解决
    检查剩余步骤，输出“预计还需 X 次训练” 
    记录每次的 step、耗时、loss 到日志文件
    

### Roadmap
+ API部署
+ Fultter部署
+ 加入评级系统
> “浓度评分”更适合在 生成完成后做自动评分或筛选，而不是一开始就塞进训练目标中
+ 加入文字加粗等
+ 情节控制机制
+ 商业化初期 若训练资源允许，可尝试知识蒸馏 + 链式优化，打造轻量高效模型
+ 商业化后期 构建自己的“日语学习大模型”，支持输入 JLPT 级别 + 输出个性化文本
