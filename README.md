# SNN Dynamic Temporal Pruning with TinyController

Spiking Neural Network với **TinyController** để dynamic timestep skipping - giảm computational cost bằng cách bỏ qua các timesteps không quan trọng.

##  Ý tưởng chính

Thay vì xử lý tất cả timesteps, sử dụng một mạng nơ-ron nhỏ (TinyController) để quyết định **skip** hay **process** từng timestep dựa trên hidden state hiện tại.

```
Hidden State [N, 512, H, W]
        ↓
  TinyController (~16K params)
        ↓
  Decision: Skip (0) or Process (1)
```

##  Cấu trúc project

```
├── archs/
│   ├── tiny_controller.py              # TinyController module
│   ├── cifar10dvs/
│   │   ├── VGGSNN.py                   # Original VGGSNN
│   │   └── VGGSNN_with_controller.py   # VGGSNN + TinyController
│   └── ucf101/
│       └── VGGSNN_UCF101.py            # VGGSNN cho UCF101 (RGB, 101 classes)
├── train_snn_with_controller.py        # Training script cho CIFAR10-DVS
├── train_snn_ucf101.py                 # Training script cho UCF101
├── kaggle_snn_ucf101.ipynb             # Kaggle notebook
└── README.md
```

##  Cách chạy

### CIFAR10-DVS (auto-download)

```bash
python train_snn_with_controller.py \
    --dataset cifar10dvs \
    --timestep 10 \
    --batch_size 16 \
    --end_iter 300
```

### UCF101 (trên Kaggle)

1. Add dataset [UCF101](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition) vào notebook
2. Upload code hoặc clone từ GitHub
3. Chạy:

```bash
python train_snn_ucf101.py \
    --data_dir "/kaggle/input/ucf101/UCF101/UCF-101" \
    --annotation_path "/kaggle/input/ucf101/UCF101TrainTestSplits-Re/ucfTrainTestlist" \
    --frames_per_clip 16 \
    --batch_size 8 \
    --epochs 50
```

##  Cấu hình

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `--target_skip` | 0.3 | Target skip ratio (30%) |
| `--frames_per_clip` | 16 | Số timesteps |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 50 | Số epochs |

##  Kết quả mong đợi

| Metric | Target |
|--------|--------|
| Skip ratio | ~30% timesteps |
| Accuracy drop | < 1% so với baseline |
| FLOPs reduction | ~30% |

##  Yêu cầu

```bash
pip install torch torchvision spikingjelly tensorboard
```

##  Tham khảo

- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) - SNN framework
- [Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Differentiable discrete sampling
