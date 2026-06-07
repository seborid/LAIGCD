#!/bin/bash
# LAIGCD 训练脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(dirname "$0")/.."

# 默认配置
DATA_PATH="data"
OUTPUT_DIR="checkpoints"
BATCH_SIZE=16
EPOCHS=30
LR=1e-4
NUM_PROTOTYPES=16
USE_FREQ=true

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --num_prototypes)
            NUM_PROTOTYPES="$2"
            shift 2
            ;;
        --no_freq)
            USE_FREQ=false
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 打印配置
echo "=========================================="
echo "LAIGCD 训练配置"
echo "=========================================="
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "学习率: $LR"
echo "原型数量: $NUM_PROTOTYPES"
echo "使用频域特征: $USE_FREQ"
echo "=========================================="
echo ""

# 运行训练
python scripts/train.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --num_prototypes "$NUM_PROTOTYPES" \
    --use_freq "$USE_FREQ" \
    --accumulation_steps 2 \
    --use_amp \
    --use_ema \
    --print_freq 50 \
    --save_freq 5
