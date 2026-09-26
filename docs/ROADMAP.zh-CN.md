# 项目规划：HandsFree（速成版）

> 用普通摄像头实现免接触电脑控制：手势鼠标 + 可训练的手势快捷键。

目标：2–3 天做出一个能放上 GitHub 和实习简历的版本。代码主要由 AI 协助编写，你的时间花在测试、采集数据、录演示和理解代码上。

## 取舍原则

只做三件事，并且每件都要能在面试里讲清楚：

1. **交互手感做对**：解决教程版的连点、抖动、远近失效问题（工程能力）
2. **一个小型 ML 模块**：自己采集数据、训练手势分类器（ML 能力）
3. **有数字、有演示**：一张 Benchmark 表 + 一段 GIF（展示能力）

## 保留 / 砍掉

| 保留 | 砍掉 |
| --- | --- |
| 直接用 MediaPipe Tasks，去掉 cvzone | 动态手势（挥手、画圈）和 GRU |
| One Euro Filter 平滑 | few-shot 自定义手势 |
| 按手掌尺寸归一化的捏合判断 + 迟滞 | ONNX 导出 |
| 手势状态机：单击、拖拽、右键、滚动 | 菜单栏应用、校准流程、停留点击 |
| 静态手势分类器（scikit-learn），绑定快捷键 | 多人数据集（有朋友帮忙就加，没有就算了） |
| 一个小 YAML 配置文件 | 复杂的多模式切换 |
| Benchmark 脚本：抖动对比、FPS、延迟 | 完整测试覆盖，只测滤波器和状态机 |
| README + 演示 GIF | |

用 scikit-learn 而不是 PyTorch：训练只要几秒，没有 Python 3.14 的兼容风险，而且对 63 维关键点特征来说完全够用。

## 目录结构（精简）

```
handsfree/
├── pyproject.toml
├── README.md
├── config.yaml                 # 摄像头、阈值、手势 → 快捷键映射
├── handsfree/
│   ├── __main__.py             # python -m handsfree
│   ├── app.py                  # 主循环
│   ├── tracker.py              # MediaPipe Tasks 封装
│   ├── filters.py              # One Euro Filter
│   ├── gestures.py             # 捏合判断 + 状态机
│   ├── classifier.py           # 特征归一化 + 加载模型推理
│   └── actions.py              # 鼠标 / 键盘控制
├── tools/
│   ├── record.py               # 采集带标签的手势数据
│   ├── train.py                # 训练并输出准确率、混淆矩阵
│   └── benchmark.py            # 抖动 / FPS / 延迟
├── tests/                      # 滤波器和状态机的单元测试
└── assets/demo.gif
```

## 三天计划

### 第 1 天：核心重构（约 4 小时）
- git 初始化、`.gitignore`、`pyproject.toml`，整理依赖（删掉 cvzone 和重复的 opencv）
- 改用 MediaPipe Tasks，按上面的结构拆分代码
- One Euro Filter、归一化捏合 + 迟滞、状态机（单击 / 拖拽 / 右键 / 滚动）
- **你要做的**：实际试用，反馈手感，调阈值

### 第 2 天：手势分类器（约 4 小时）
- `record.py`：按数字键打标签采集关键点
- 5–6 个静态手势，例如：握拳 = 暂停控制，手掌 = 恢复，点赞 = 播放/暂停，V 字 = 截图，OK = 回车
- 特征：以手腕为原点、按手掌尺寸缩放，左手镜像
- `train.py`：对比 2–3 个模型（如 KNN、随机森林、MLP），输出准确率和混淆矩阵图
- 在主程序里加入分类器，识别结果需要连续多帧一致才触发，防止误触发
- **你要做的**：每个手势采集 1–2 分钟数据（换角度、换距离、换光线）

### 第 3 天：评测和包装（约 4 小时）
- `benchmark.py`：手静止时光标抖动（原始 vs 指数平滑 vs One Euro）、FPS、单帧处理延迟
- 几个单元测试 + 简单的 GitHub Actions
- README：GIF 放最前面，然后是功能、架构图、安装、Benchmark 表、已知限制
- **你要做的**：录演示 GIF，通读代码，确保每个设计你都能解释

## 面试时要能讲清楚的点

- 为什么原版会连点，状态机怎么解决的
- 为什么用 One Euro Filter 而不是固定系数平滑（速度自适应：慢时强平滑，快时低延迟）
- 为什么捏合阈值要按手掌尺寸归一化
- 分类器的特征怎么设计，为什么要做平移和缩放归一化，怎么防止误触发
- Benchmark 的数字是怎么测的

## 简历条目模板（数字以实测为准）

> **HandsFree** — Touchless computer control via hand gestures | Python, MediaPipe, OpenCV, scikit-learn
> - Built a real-time gesture control system with a modular pipeline (tracking → filtering → recognition → actions), supporting click, drag, scroll, and configurable gesture shortcuts.
> - Reduced cursor jitter by X% vs. naive smoothing using a One Euro filter, and eliminated repeated false clicks with a hysteresis-based gesture state machine.
> - Collected an X-sample hand landmark dataset and trained a gesture classifier achieving X% test accuracy at X ms per frame.
