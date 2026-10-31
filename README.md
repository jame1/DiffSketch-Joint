# DiffSketch-Joint

端到端双扩散敦煌壁画修复参考实现。形状约定：图像 `[-1,1]`，线描概率 `[0,1]`，掩码 `1=破损`。

## 安装

```bash
pip install torch torchvision pillow diffusers transformers accelerate
```

首次加载 `stabilityai/sd-vae-ft-mse` 需要可访问 Hugging Face 并下载权重；离线或失败时会警告并使用仅用于调试的随机冻结 fallback VAE（不是预训练 VAE）。

## 数据目录

```text
data/
  images/   # 完整 RGB 图像
  sketches/ # 单通道线描图，同名配对
  masks/    # 单通道破损掩码，同名配对，白色为破损
```

## 三阶段训练

```bash
python train.py --data ./data --stage 1 --epochs 20 --batch-size 2
python train.py --data ./data --stage 2 --epochs 20 --batch-size 2 --resume checkpoints/stage1.pt
python train.py --data ./data --stage 3 --epochs 10 --batch-size 1 --resume checkpoints/stage2.pt
```

`--resume` 用于恢复权重和优化器状态；若从不同 stage 切换，优化器状态结构可能不兼容，可不传 `--resume`，先加载相应阶段模型权重的逻辑需按实际实验配置调整。

## 推理

```python
from PIL import Image
from infer import DiffSketchJoint
model = DiffSketchJoint()
model.load_checkpoint("checkpoints/stage3.pt")
result = model.infer(Image.open("damaged.png"), Image.open("mask.png"), num_steps=50)
result.save("restored.png")
```

## 重要实现说明

这是可运行的研究原型，不是与原版 Stable Diffusion U-Net 完全等价的 LDM：VAE 可使用预训练 SD VAE，扩散去噪 U-Net 为轻量自定义网络，以便在普通 GPU 上训练。用于正式实验前应替换为与 VAE latent scaling / scheduler 匹配的 diffusers UNet2DConditionModel，并验证数据尺寸、显存、损失数值及恢复质量。当前结构调制为可微的 sigmoid 残差门控；推理循环为简化的确定性 DDPM/DDIM 风格采样。
