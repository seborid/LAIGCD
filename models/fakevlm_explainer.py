"""
FakeVLM 解释器模块

集成LLaVA-1.5模型用于生成AIGC检测的自然语言解释
支持零样本推理和预训练模型加载
"""

import torch
import warnings
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from PIL import Image


class FakeVLMExplainer:
    """
    FakeVLM解释器 - 基于LLaVA的多模态解释生成

    功能：
        - 接收原图+热力图，生成自然语言解释
        - 支持零样本推理
        - 支持预训练模型加载

    Args:
        model_path: 模型路径（本地或HuggingFace）
        device: 运行设备
        load_in_8bit: 是否8bit量化（节省显存）
        use_flash_attn: 是否使用Flash Attention 2
    """

    def __init__(
        self,
        model_path: str = "lingcco/fakeVLM",
        device: str = "cuda",
        load_in_8bit: bool = True,
        use_flash_attn: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.use_flash_attn = use_flash_attn

        self.model = None
        self.processor = None
        self._is_loaded = False
        self._device_map_set = False  # 标记是否使用了device_map

    def load_model(self):
        """加载FakeVLM模型（懒加载）"""
        if self._is_loaded:
            return

        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError:
            raise ImportError(
                "需要安装 transformers: pip install transformers>=4.45.0"
            )

        print(f"正在加载 FakeVLM 模型: {self.model_path}")

        # 加载processor
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # 模型加载参数
        loading_kwargs = {
            "low_cpu_mem_usage": True,
        }

        # 8bit量化 - 使用bitsandbytes
        if self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.bfloat16,
                    bnb_8bit_use_double_quant=True,
                )
                loading_kwargs["quantization_config"] = quantization_config
                loading_kwargs["device_map"] = "auto"
                self._device_map_set = True
                print("使用 8bit 量化加载")
            except ImportError:
                warnings.warn("bitsandbytes未安装，8bit量化不可用，将使用fp16")
                self.load_in_8bit = False

        # 数据类型（仅在非8bit时设置）
        if not self.load_in_8bit:
            if self.device == "cuda":
                loading_kwargs["torch_dtype"] = torch.float16
                print("使用 fp16 加载")
            else:
                loading_kwargs["torch_dtype"] = torch.float32
                print("使用 fp32 加载")

        # Flash Attention
        if self.use_flash_attn:
            try:
                loading_kwargs["use_flash_attention_2"] = True
                print("启用 Flash Attention 2")
            except Exception as e:
                warnings.warn(f"Flash Attention 2不可用: {e}")

        # 加载模型
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            **loading_kwargs
        )

        self.model.eval()

        # 手动移动模型到设备（仅当没有使用device_map时）
        if not self._device_map_set and self.device == "cuda":
            self.model = self.model.to(self.device)

        print(f"✓ FakeVLM 模型加载成功")
        self._is_loaded = True

    def _to_pil_image(
        self,
        img_input: Union[Image.Image, torch.Tensor, str, np.ndarray],
        is_heatmap: bool = False,
        cmap: str = "jet",
    ) -> Image.Image:
        """将各种格式的输入转换为PIL Image"""
        if isinstance(img_input, str):
            # 文件路径
            return Image.open(img_input).convert("RGB")

        elif isinstance(img_input, Image.Image):
            # 已经是PIL Image
            return img_input.convert("RGB")

        elif isinstance(img_input, np.ndarray):
            # numpy数组
            if is_heatmap:
                return self._numpy_to_pil(img_input, cmap=cmap)
            else:
                # 原图，假设范围是[0,1]或[0,255]
                if img_input.max() <= 1.0:
                    img_input = (img_input * 255).astype("uint8")
                return Image.fromarray(img_input).convert("RGB")

        elif isinstance(img_input, torch.Tensor):
            # torch tensor
            if is_heatmap:
                return self._tensor_to_pil(img_input, cmap=cmap)
            else:
                # 原图
                if img_input.dim() == 3:
                    img_input = img_input.permute(1, 2, 0)
                img_array = img_input.detach().cpu().numpy()
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype("uint8")
                return Image.fromarray(img_array).convert("RGB")

        else:
            raise TypeError(f"不支持的输入类型: {type(img_input)}")

    def _tensor_to_pil(self, tensor: torch.Tensor, cmap: str = "jet") -> Image.Image:
        """将热力图tensor转为PIL Image"""
        import matplotlib.pyplot as plt
        from io import BytesIO

        if tensor.dim() == 3:
            tensor = tensor[0]

        heatmap = tensor.detach().cpu().numpy()
        heatmap = np.clip(heatmap, 0, 1)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(heatmap, cmap=cmap, vmin=0, vmax=1)
        ax.axis("off")
        plt.tight_layout(pad=0)

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf).convert("RGB")

    def _numpy_to_pil(self, array: np.ndarray, cmap: str = "jet") -> Image.Image:
        """将numpy数组转为PIL Image"""
        import matplotlib.pyplot as plt
        from io import BytesIO

        array = np.clip(array, 0, 1)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(array, cmap=cmap, vmin=0, vmax=1)
        ax.axis("off")
        plt.tight_layout(pad=0)

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf).convert("RGB")

    def generate_explanation(
        self,
        original: Union[Image.Image, torch.Tensor, str, np.ndarray],
        spatial_heatmap: Optional[Union[Image.Image, torch.Tensor, str, np.ndarray]] = None,
        frequency_heatmap: Optional[Union[Image.Image, torch.Tensor, str, np.ndarray]] = None,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
        top_prototypes: Optional[List[Dict]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> Dict[str, Any]:
        """
        生成自然语言解释

        注意：FakeVLM/LLaVA 主要是为单图设计的，多图支持可能不稳定
        因此这里只使用原图，其他参数仅用于兼容性

        Args:
            original: 原图
            prediction: 检测结果 ("Real" 或 "Fake")
            confidence: 置信度 (0-1)
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            do_sample: 是否采样

        Returns:
            dict: 包含解释文本和元数据
        """
        if not self._is_loaded:
            self.load_model()

        # 只处理原图（FakeVLM 主要为单图设计）
        original_pil = self._to_pil_image(original, is_heatmap=False)

        # 构建prompt
        prompt = self._build_prompt(
            prediction=prediction,
            confidence=confidence,
        )

        # 处理输入
        inputs = self.processor(
            text=prompt,
            images=original_pil,
            return_tensors="pt",
            padding=True,
        )

        # 修复image tokens数量不匹配的问题
        # LLaVA模型期望576个image tokens (24x24 patches)，但processor只插入575个
        # 需要手动添加一个额外的<image> token
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        input_ids = inputs["input_ids"][0]
        image_token_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]

        if len(image_token_indices) == 575:
            # 需要添加一个额外的<image> token
            # 在最后一个<image> token后面插入
            last_idx = image_token_indices[-1].item()
            new_input_ids = torch.cat([
                input_ids[:last_idx + 1],
                torch.tensor([image_token_id], dtype=input_ids.dtype),
                input_ids[last_idx + 1:]
            ])
            inputs["input_ids"] = new_input_ids.unsqueeze(0)
            # 同时需要更新attention_mask（如果存在）
            if "attention_mask" in inputs:
                attention_mask = inputs["attention_mask"][0]
                new_attention_mask = torch.cat([
                    attention_mask[:last_idx + 1],
                    torch.tensor([1], dtype=attention_mask.dtype),
                    attention_mask[last_idx + 1:]
                ])
                inputs["attention_mask"] = new_attention_mask.unsqueeze(0)

        # 移到设备
        target_device = self.device if not self._device_map_set else "cuda"
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # 准备生成参数
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }

        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_p"] = 0.9
            gen_kwargs["top_k"] = 50
        else:
            gen_kwargs["do_sample"] = False

        # 生成
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # 解码
        explanation = self.processor.decode(output_ids[0], skip_special_tokens=False)

        # 提取生成的部分（去除prompt）
        if prompt in explanation:
            explanation = explanation.split(prompt)[-1].strip()

        # 移除可能的特殊token
        explanation = explanation.replace("<s>", "").replace("</s>", "").strip()

        return {
            "explanation": explanation,
            "prompt": prompt,
            "num_images": 1,  # 只使用原图
            "model_path": self.model_path,
        }

    def _build_prompt(
        self,
        prediction: Optional[str],
        confidence: Optional[float],
    ) -> str:
        """构建推理prompt（LLaVA格式）"""

        # 图像占位符（单图）
        image_placeholder = "<image>\n"

        # 检测结果信息
        detection_info = ""
        if prediction and confidence:
            detection_info = f"\n初步检测结果: {prediction} (置信度: {confidence:.1%})"

        # 完整prompt - 使用 LLaVA 格式
        prompt = f"""USER: {image_placeholder}
你是AI生成人脸检测专家。请分析这张人脸图像：

{detection_info}

请从以下角度分析：
1. 面部特征自然度（发际线、瞳孔、眉毛、光影、皮肤纹理等）
2. 是否存在AI生成的典型伪影（模糊、不协调、不对称等）
3. 综合判断是否为AI生成的人脸图像，并详细说明依据

请给出清晰的分析结论和理由。
ASSISTANT:
"""

        return prompt

    def batch_explain(
        self,
        items: List[Dict[str, Any]],
        max_new_tokens: int = 256,
    ) -> List[Dict[str, Any]]:
        """
        批量生成解释

        Args:
            items: 包含原始图像和热力图的字典列表
            max_new_tokens: 最大生成token数

        Returns:
            包含解释的结果列表
        """
        results = []
        for item in items:
            result = self.generate_explanation(
                original=item.get("original"),
                spatial_heatmap=item.get("spatial_heatmap"),
                frequency_heatmap=item.get("frequency_heatmap"),
                prediction=item.get("prediction"),
                confidence=item.get("confidence"),
                top_prototypes=item.get("top_prototypes"),
                max_new_tokens=max_new_tokens,
            )
            results.append({**item, "explanation": result})
        return results

    def unload_model(self):
        """卸载模型释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._is_loaded = False
        self._device_map_set = False

        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("✓ FakeVLM 模型已卸载")


def create_explainer(
    model_path: Optional[str] = None,
    device: str = "cuda",
    load_in_8bit: bool = True,
) -> FakeVLMExplainer:
    """
    创建FakeVLM解释器的工厂函数

    Args:
        model_path: 模型路径（默认使用lingcco/fakeVLM）
        device: 运行设备
        load_in_8bit: 是否8bit量化

    Returns:
        FakeVLMExplainer实例
    """
    if model_path is None:
        model_path = "lingcco/fakeVLM"

    return FakeVLMExplainer(
        model_path=model_path,
        device=device,
        load_in_8bit=load_in_8bit,
    )


def test_explainer():
    """测试FakeVLM解释器"""
    print("测试 FakeVLM 解释器...")

    # 创建解释器
    explainer = create_explainer()

    # 测试加载
    explainer.load_model()

    print("✓ FakeVLM 解释器测试完成")


if __name__ == "__main__":
    test_explainer()
