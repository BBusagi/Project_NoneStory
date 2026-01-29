# NoneStory - 日语 N1 学习辅助生成式 AI

**项目复盘（简要）**  
**当前状态：暂停**  
**难点分析：** 如何提升客户端响应速度，尤其是移动端体验  
**学习到的内容：**  
1. 文本 LLM 在训练成本、推理延迟与质量控制上挑战更突出  
2. 数据清洗与质量控制非常关键  
3. 硬件资源要求较高（显存、稳定性、训练时间）

## 项目简介
面向日语 N1 单词学习的定制语言模型项目。通过训练后的文生文模型生成现代小说故事，帮助用户在阅读中学习和记忆 N1 词汇，实现“故事体验 + 学习”并存。

## Demo / 模型
- HuggingFace Space: https://huggingface.co/spaces/BBusagi/sft-nonestory-generator
- 模型（train002）: https://huggingface.co/BBusagi/sft-nonestory-train002
- 模型（train001）: https://huggingface.co/BBusagi/sft-nonestory-train001

## 环境说明
- Python 3.10
- 训练环境: `conda activate rinna-env`
- GPU 监控: `nvidia-smi -l 1`
- Unicode 处理参考: https://blog.cloudnative.co.jp/23733/

## 当前状态
- [Bug] 网页部署输出无法访问
- [Task] 模型输出不稳定
- [Task] 手机 / 低配置电脑长时间无响应

## 系统设计
1. 抽取 JLPT N1 必备词汇
2. 用 GPT-4 生成高质量语料
3. 构建训练数据集
4. 训练（全参微调）rinna-medium

## 开发流程
### Step00 环境配置和检测
- Script/Tool_check_api_connection.py

### Step01 准备语料
- Script/Step1_generate_json.py
- Script/Tool_convert_jsonl_to_txt.mjs
- Script/Step1_prepare_training_data.py

### Step02 训练模型（含断点保存）
- Script/Tool_check_gpu_status.py
- Script/Step2_train_sft_model.py
- Script/Step2_save_model.py

### Step03 模型评估
- Script/Step3_infer_and_evaluate.py
- Script/Tool_run_model_inference.py
- Script/Step3_compare_multiple_models.py

## Dev Log
### Step01: chatGPT + rinna/small
- 环境配置
- OpenAI / rinna 测试
- Batch 调用与质量检查（生成 1000 篇）
- Prompt 调整
- 语料预处理与训练 JSON 生成
- SFT 训练尝试
- 结果：可行，模仿语料成功；长度控制较弱，整体较浅

### Step02: GPT-4 长文语料 + rinna/medium
- Prompt 与 Batch 更新
- BUG: 90,000 enqueued tokens 限制（4o -> 4omini）
- 语料预处理与训练 JSON 生成
- SFT 训练测试
- BUG: 全参调节长度不一致（参数提纯）
- 全参调节训练（train2）
- BUG: 蓝屏 / 显存溢出（更换 pytorch 调用 GPU 方式）
- 断点保存、多线程开启
- 单次训练约 90 分钟
- checkpoint-2880：表现优秀，生成内容符合标准  
  ![](readme/train1(2880).png)
- 断点续练
- BUG: lm_head 缺失权重文件（pytorch_model.bin）
  - 调整保存模块 + 小片段训练（尝试 LoRA），仍存在
  - 原因：resume_from_checkpoint 旧版仅支持 .bin，新版为 .safetensors
  - 处理：移除 resume_from_checkpoint 参数（已解决）
- 全参调节训练（train1）
- checkpoint-1500：
  - 第一次：模型状态回滚，生成长度不足，效果极差
  - 第二次：表现优秀，长度已增长  
  ![](readme/train2(1500).png)

### Step03: 前端部署
- 模型封装仓库
- HuggingFace Space 搭建
- Gradio 前端部署
- 页面修改与功能增强
  - 生成历史查看
  - 多模型选择
  - 批量生成
  - 异常处理
  - 手动删除

## TODO
### 模型
- 训练监控与完整性检测模块
- 输出剩余训练次数（预计还需 X 次）
- 记录 step / 耗时 / loss 到日志文件

### 评价系统
- 加入评级系统  
  备注：浓度评分更适合生成完成后进行自动评分或筛选，而不是加入训练目标
- 支持文字加粗等格式
- 情节控制机制
- 后期：如资源允许，尝试知识蒸馏 + 链式优化，构建轻量模型
- 后期：构建可输入 JLPT 级别的“日语学习大模型”，输出个性化文本

### 前端
- 流式生成（实时输出）
- 任务队列系统（多人访问时轮询生成）
