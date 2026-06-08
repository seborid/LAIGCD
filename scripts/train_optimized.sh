#!/bin/bash
# LAIGCD 优化的训练脚本 - RTX 4070 Ti (12GB)

# ================================================================
# 硬件配置
# GPU: RTX 4070 Ti (12GB VRAM, ~40 TFLOPS FP32)
# 内存: 8GB RAM
# ================================================================

# ================================================================
# 训练时间估算
# ================================================================
# 模式         | 样本数  | 轮数 | 预估时间 | 用途
#--------------|---------|------|----------|------
# debug        | 100     | 3    | ~3min    | 验证代码
# quick        | 1,000   | 5    | ~10min   | 快速测试
# standard     | 10,000  | 15   | ~2h      | 中等测试
# full         | ~270K   | 30   | ~8h      | 完整训练
# ================================================================

set -e  # 遇到错误退出

# ================================================================
# 配置参数
# ================================================================

# 训练模式: debug | quick | standard | full
MODE="${1:-quick}"

# 数据集路径
DATA_PATH="data"

# 输出目录
OUTPUT_DIR="checkpoints/${MODE}_run"

# ================================================================
# 根据模式设置参数
# ================================================================

case $MODE in
    debug)
        # 调试模式：快速验证代码能跑通
        MAX_SAMPLES=100
        EPOCHS=3
        BATCH_SIZE=64
        SAVE_FREQ=1
        ;;
    quick)
        # 快速测试：验证模型能收敛
        MAX_SAMPLES=1000
        EPOCHS=5
        BATCH_SIZE=64
        SAVE_FREQ=2
        ;;
    standard)
        # 标准测试：中等规模，验证效果
        MAX_SAMPLES=10000
        EPOCHS=15
        BATCH_SIZE=64
        SAVE_FREQ=5
        ;;
    full)
        # 完整训练：使用全部数据
        MAX_SAMPLES=""
        EPOCHS=30
        BATCH_SIZE=64
        SAVE_FREQ=5
        ;;
    *)
        echo "未知模式: $MODE"
        echo "支持模式: debug, quick, standard, full"
        exit 1
        ;;
esac

# ================================================================
# 固定参数（针对4070 Ti优化）
# ================================================================

# 数据加载
NUM_WORKERS=4          # 数据加载线程数
IMG_SIZE=224           # CLIP输入大小

# 模型参数
CLIP_MODEL="ViT-B-32"
NUM_PROTOTYPES=16
USE_FREQ=true
FREQ_TYPE="srm"
DROPOUT=0.1

# 训练参数
LR=1e-4
WEIGHT_DECAY=0.01
WARMUP_EPOCHS=2
CLIP_GRAD_NORM=1.0

# 优化选项
USE_AMP=true          # 混合精度训练（节省显存，加速约40%）
USE_EMA=true          # EMA（提升稳定性）
EMA_DECAY=0.9999

# ================================================================
# 打印配置
# ================================================================

echo "=================================================="
echo "LAIGCD 训练 - RTX 4070 Ti 优化配置"
echo "=================================================="
echo "模式: $MODE"
echo "样本数: $MAX_SAMPLES"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "输出目录: $OUTPUT_DIR"
echo "数据集路径: $DATA_PATH"
echo "=================================================="

# ================================================================
# 训练命令
# ================================================================

# 构建命令（根据MAX_SAMPLES是否为空）
CMD="python scripts/train.py \
    --data_path \"$DATA_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --img_size $IMG_SIZE \
    --clip_model \"$CLIP_MODEL\" \
    --num_prototypes $NUM_PROTOTYPES \
    --use_freq \
    --freq_type \"$FREQ_TYPE\" \
    --dropout $DROPOUT \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_epochs $WARMUP_EPOCHS \
    --clip_grad_norm $CLIP_GRAD_NORM \
    --use_amp \
    --use_ema \
    --ema_decay $EMA_DECAY \
    --save_freq $SAVE_FREQ \
    --print_freq 50 \
    --device cuda"

# 如果设置了MAX_SAMPLES，添加该参数
if [ -n "$MAX_SAMPLES" ]; then
    CMD="--max_samples $MAX_SAMPLES $CMD"
fi

# 执行命令
eval $CMD

echo ""
echo "=================================================="
echo "训练完成！"
echo "模型保存在: $OUTPUT_DIR/best_model.pth"
echo "=================================================="
