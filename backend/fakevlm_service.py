#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FakeVLM 解释服务模块

作为两阶段检测流水线的第二阶段：第一阶段（DetectionService，CLIP 分类器）给出
Real/Fake 判定，本服务被告知该判定后，由 FakeVLM（LLaVA-1.5-7B 微调模型）输出
自然语言理由 / 伪造痕迹解释。

加载与推理逻辑改编自 FakeVLM/scripts/infer.py（独立仓库，不直接 import）。
"""

import os
import json
import logging
from typing import Optional, Dict, Any

import torch
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置（模型权重位于 LAIGCD 仓库之外，通过环境变量覆盖）
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_FVLM_ROOT = os.path.join(_PROJECT_ROOT, "..", "FakeVLM", "models")

DEFAULT_MODEL_PATH = os.environ.get(
    "FAKEVLM_MODEL_PATH",
    os.path.abspath(os.path.join(_DEFAULT_FVLM_ROOT, "lingcco_fakeVLM")),
)
DEFAULT_PROCESSOR_PATH = os.environ.get(
    "FAKEVLM_PROCESSOR_PATH",
    os.path.abspath(os.path.join(_DEFAULT_FVLM_ROOT, "llava-hf_llava-1.5-7b-hf")),
)
DEFAULT_DEVICE = os.environ.get("FAKEVLM_DEVICE", "cuda")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("FAKEVLM_MAX_TOKENS", "256"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# 默认开启 8-bit 量化：12GB 显存下 7B FP16（~14GB）无法容纳。
DEFAULT_USE_8BIT = _env_bool("FAKEVLM_8BIT", True)
DEFAULT_USE_FLASH_ATTN = _env_bool("FAKEVLM_FLASH_ATTN", True)
# 量化模式："8bit"（线性 8-bit）或 "4bit"（NF4 4-bit，显存更省、常保真度更好）。
DEFAULT_QUANT_MODE = os.environ.get("FAKEVLM_QUANT", "8bit").strip().lower()

# ---------------------------------------------------------------------------
# 修改后的 prompt（本任务核心）
#
# 与 infer.py 默认的 "Does the image looks real/fake? give me 3 reasons." 不同，
# 这里直接「告诉模型图片的真假性」（取自第一阶段判定），让模型输出理由。
# 措辞保留训练时的 "give me 3 reasons" 以贴近训练分布，保证解释质量。
# ---------------------------------------------------------------------------
PROMPT_FAKE = (
    "This image is fake (AI-generated). Look at the image carefully and explain "
    "why it is fake, describing the specific visual artifacts that reveal it is "
    "synthetic. Give me 3 reasons."
)

PROMPT_REAL = (
    "This image is real. Look at the image carefully and explain why it appears to "
    "be a genuine, non-synthetic photograph. Give me 3 reasons."
)


def configure_processor(processor, model):
    """回填新版 transformers 所需的 LLaVA processor 字段。

    直接改编自 FakeVLM/scripts/infer.py 的 configure_processor。
    """
    vision_config = getattr(model.config, "vision_config", None)

    if getattr(processor, "patch_size", None) is None and vision_config is not None:
        processor.patch_size = getattr(vision_config, "patch_size", None)

    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = getattr(
            model.config,
            "vision_feature_select_strategy",
            None,
        )

    processor_config_path = None
    if isinstance(getattr(processor, "name_or_path", None), str):
        candidate = os.path.join(processor.name_or_path, "processor_config.json")
        if os.path.exists(candidate):
            processor_config_path = candidate

    loaded_num_additional_tokens = None
    if processor_config_path is not None:
        try:
            with open(processor_config_path, "r", encoding="utf-8") as f:
                loaded_num_additional_tokens = json.load(f).get(
                    "num_additional_image_tokens"
                )
        except Exception:
            loaded_num_additional_tokens = None

    if loaded_num_additional_tokens is None:
        # CLIP vision tower 前置一个 CLS token，LLaVA 在应用 "default" 特征选择规则前
        # 需要一个额外的 image token。
        processor.num_additional_image_tokens = 1
    else:
        processor.num_additional_image_tokens = loaded_num_additional_tokens

    missing_fields = []
    if getattr(processor, "patch_size", None) is None:
        missing_fields.append("patch_size")
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        missing_fields.append("vision_feature_select_strategy")

    if missing_fields:
        raise ValueError(
            "Processor is missing required LLaVA fields after configuration: "
            + ", ".join(missing_fields)
        )


class FakeVLMService:
    """
    FakeVLM 解释服务

    负责加载 FakeVLM 模型并基于第一阶段判定生成自然语言理由。
    采用懒加载：首次调用 explain() 时才加载 7B 模型。
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        processor_path: str = DEFAULT_PROCESSOR_PATH,
        device: str = DEFAULT_DEVICE,
        use_flash_attn: bool = DEFAULT_USE_FLASH_ATTN,
        load_in_8bit: bool = DEFAULT_USE_8BIT,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        quant_mode: str = DEFAULT_QUANT_MODE,
    ):
        self.model_path = model_path
        self.processor_path = processor_path
        self.device = device
        self.use_flash_attn = use_flash_attn
        self.load_in_8bit = load_in_8bit
        self.max_new_tokens = max_new_tokens
        self.quant_mode = quant_mode if quant_mode in ("8bit", "4bit") else "8bit"

        self.model = None
        self.processor = None

    def _load_model(self):
        """加载 FakeVLM 模型与 processor（改编自 infer.py 的 load_model）。"""
        logger.info(f"正在加载 FakeVLM 模型: {self.model_path}")

        device = self.device
        use_flash_attn = self.use_flash_attn
        load_in_8bit = self.load_in_8bit

        # CUDA 可用性检查
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("请求 cuda 但不可用，回退到 CPU。")
            device = "cpu"

        if device == "cpu":
            logger.warning("使用 CPU 推理，速度会非常慢。")
            use_flash_attn = False
            if load_in_8bit:
                logger.warning("CPU 不支持 8-bit 量化，已禁用。")
                load_in_8bit = False

        # processor 路径：若模型目录下有 processor_config.json 则优先用模型目录
        processor_path = (
            self.model_path
            if os.path.exists(os.path.join(self.model_path, "processor_config.json"))
            else self.processor_path
        )
        processor = AutoProcessor.from_pretrained(processor_path)

        loading_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }

        # 量化配置。12GB 显存必须量化；bf16 compute 在当前 bitsandbytes + transformers 下会
        # 生成空文本，因此统一使用 fp16 compute dtype。
        do_quant = (device != "cpu") and (load_in_8bit or self.quant_mode == "4bit")
        if do_quant:
            if self.quant_mode == "4bit":
                loading_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("已启用 4-bit NF4 量化（显存约 5GB，compute fp16）")
            else:
                loading_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True,
                )
                logger.info("已启用 8-bit 量化（显存约 7GB，compute fp16）")
            loading_kwargs["torch_dtype"] = None  # 量化时由 quantization_config 决定

        if device == "cuda":
            loading_kwargs["device_map"] = "auto"
            logger.info(
                f"使用 GPU: {torch.cuda.get_device_name(0)}，"
                f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
        elif device == "auto":
            loading_kwargs["device_map"] = "auto"
        else:  # cpu
            loading_kwargs["device_map"] = None

        if use_flash_attn and device == "cuda":
            try:
                import flash_attn  # noqa: F401  仅当包存在时才启用
                loading_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("已启用 Flash Attention 2")
            except Exception as e:
                logger.warning(
                    f"未启用 Flash Attention 2（{e}），将使用默认注意力实现"
                )

        logger.info("正在加载模型权重（首次运行可能较慢）...")
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            **loading_kwargs,
        ).eval()

        configure_processor(processor, model)

        self.model = model
        self.processor = processor
        logger.info("FakeVLM 模型加载完成")

    def _select_prompt(self, verdict: str) -> str:
        """根据第一阶段判定选择对应 prompt（不区分大小写，默认按 fake 处理）。"""
        v = (verdict or "").strip().lower()
        if v.startswith("real"):
            return PROMPT_REAL
        return PROMPT_FAKE

    def explain(
        self,
        image: Image.Image,
        verdict: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """基于第一阶段判定生成自然语言理由。

        Args:
            image: PIL Image 对象（RGB）
            verdict: 第一阶段判定，"Real" 或 "Fake"
            max_new_tokens: 最大生成 token 数

        Returns:
            模型生成的理由文本（英文）。
        """
        if self.model is None:
            self._load_model()

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        prompt = self._select_prompt(verdict)

        # 用模型的 chat template 格式化输入，以便正确插入 image token
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # 只解码新生成的 token（去掉 prompt 部分）
        prompt_length = inputs["input_ids"].shape[-1]
        generated_tokens = output[0][prompt_length:]
        response = self.processor.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        return response

    def get_model_info(self) -> Dict[str, Any]:
        """返回模型配置信息。"""
        return {
            "model_path": self.model_path,
            "processor_path": self.processor_path,
            "device": self.device,
            "quant_mode": self.quant_mode,
            "load_in_8bit": self.load_in_8bit,
            "use_flash_attn": self.use_flash_attn,
            "max_new_tokens": self.max_new_tokens,
            "loaded": self.model is not None,
        }
