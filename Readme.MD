env
python 3.10
nvidia-smi - 查看显卡配置
conda activate rinna-env - 切换到训练环境
\\u.{4} - [URL](https://blog.cloudnative.co.jp/23733/)

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
    + 全参调节
    > BUG 系统蓝屏 显存溢出 - 重新pythor调用GPU
    + !!!TODO!!! 断点保存 多线程开启

### Roadmap
+ 加入评级系统
> “浓度评分”更适合在 生成完成后做自动评分或筛选，而不是一开始就塞进训练目标中
+ 加入文字加粗等
+ 商业化初期 若训练资源允许，可尝试知识蒸馏 + 链式优化，打造轻量高效模型
+ 商业化后期 构建自己的“日语学习大模型”，支持输入 JLPT 级别 + 输出个性化文本
