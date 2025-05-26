## note env  
python 3.10  
conda activate rinna-env - 切换到训练环境  
nvidia-smi -l 1 - 查看显卡配置  
\\u.{4} - [URL](https://blog.cloudnative.co.jp/23733/)  

## 项目介绍
语言模型定制项目，为日语N1单词学习辅助生成式AI，目的是通过特殊训练后的文生文AI生成N1的现代小说故事来辅助用户学习和记忆N1相关的知识点。

## Model & Demo
[Demo - HuggingFaceSpace](https://huggingface.co/spaces/BBusagi/sft-nonestory-generator)  
[sft-nonestory-train002](https://huggingface.co/BBusagi/sft-nonestory-train002)  
[sft-nonestory-train001](https://huggingface.co/BBusagi/sft-nonestory-train001)  

##
### TODO
[Bug]网页部署输出无法访问
[Task]模型目前输出不稳定
[Task]手机，低配置电脑长时间无响应

### Program Design
1. 抽取日语 N1等级考试所需要的单词
2. 使用 GPT-4 自动生成高质量语料
3. 构建训练用数据集
4. 全参调节模型（rinna-medium）

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
+ ✅Step01  chatGPT+rinna/small
    + 环境配置
    + OpenAPI测试和rinna测试
    + 尝试Batch调用
    + Batch检测，生成1000个故事
    + 修改prompt，到达满意阶段
    + 预处理语料，生成训练JSON
    + sft训练尝试
    + 测试：**可行 模仿语料成功 长度控制有问题 整体文章较肤浅**

+ ✅Step02  gpt4长文语料 + rinna/medium
    + prompt修改更新
    + batch更新 
    >BUG Limit: 90,000 enqueued tokens - 4o更换为4omini
    + 预处理语料，生成训练JSON
    + sft训练测试
    > BUG 全参调节 长度不一致 - 参数提纯
    + sft训练 全参调节（train2）
    > BUG 系统蓝屏 显存溢出 - 重新pythor调用GPU
    + 断点保存 多线程开启
    + 每次运行 90 分钟训练 
    + 测试：[checkpoint-2880](https://huggingface.co/BBusagi/sft-nonestory-train001) -> **表现状态优秀，生成内容符合标准**
    ![](readme/train1(2880).png)
    + 断点续练
    > BUG lm_head
    > 1. 缺失模型参数权重文件（pytorch—_model.bin）
    > -- 修改save模块 重新小片段训练 (尝试LORA) 
    > -- 仍然存在
    > 2. resume_from_checkpoint = 老版本，默认只支持.bin；新版本保存格式为.safetensors
    > -- trainer.train(~~resume_from_checkpoint~~)
    > -- ✅已解决
    + sft训练  全参调节（train1）
    + 检测：[checkpoint-1500](https://huggingface.co/BBusagi/sft-nonestory-train002) -> 
    第一次测试**模型状态回滚，生成长度不足，效果极差**
    第二次测试**表现优秀，长度已增长**
    ![](readme/train2(1500).png)

+ ✅Step03  网页前端部署
    + 模型封装仓库
    + Space搭建
    + Gradio前端部署
    + 网页修改
    + 功能 生成历史查看
    + 功能 多模型选择
    + 功能 批量生成
    + 功能 添加异常处理
    + 功能 添加手动删除


### Roadmap
#### 模型
+ 训练监控 完整性检测相关模组
+ 检查剩余步骤，输出“预计还需 X 次训练” 
+ 记录每次的 step、耗时、loss 到日志文件

#### 评价系统
+ 加入评级系统
> “浓度评分”更适合在 生成完成后做自动评分或筛选，而不是一开始就塞进训练目标中
+ 加入文字加粗等
+ 情节控制机制
+ 后期1 若训练资源允许，可尝试知识蒸馏 + 链式优化，打造轻量高效模型
+ 后期2 构建自己的“日语学习大模型”，支持输入 JLPT 级别 + 输出个性化文本

#### 前端
+ 流生成
+ 任务队列系统
