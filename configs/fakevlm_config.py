"""
FakeVLM 配置文件

管理FakeVLM模型路径和推理参数
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FakeVLMConfig:
    """FakeVLM解释器配置"""

    # 模型配置
    model_path: str = "lingcco/fakeVLM"
    device: str = "cuda"
    load_in_8bit: bool = True
    use_flash_attn: bool = False

    # 推理配置
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50

    # 输出配置
    save_nl_explanation: bool = True
    save_heatmaps: bool = True

    # 资源管理
    unload_after_use: bool = True
    batch_size: int = 1  # FakeVLM通常batch_size=1

    # 本地模型路径（如有预下载的模型）
    local_model_path: Optional[str] = None


# 预设配置
PRESET_CONFIGS = {
    "default": FakeVLMConfig(),
    "fast": FakeVLMConfig(
        max_new_tokens=128,
        temperature=0.5,
    ),
    "detailed": FakeVLMConfig(
        max_new_tokens=512,
        temperature=0.8,
        do_sample=True,
    ),
    "low_memory": FakeVLMConfig(
        load_in_8bit=True,
        max_new_tokens=128,
    ),
}


def get_config(preset: str = "default", **kwargs) -> FakeVLMConfig:
    """
    获取配置

    Args:
        preset: 预设名称 ("default", "fast", "detailed", "low_memory")
        **kwargs: 覆盖的配置项

    Returns:
        FakeVLMConfig实例
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"未知预设: {preset}，可选: {list(PRESET_CONFIGS.keys())}")

    config = PRESET_CONFIGS[preset]

    # 覆盖配置
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"未知配置项: {key}")

    return config


if __name__ == "__main__":
    # 测试配置
    config = get_config("default")
    print(f"默认配置: {config}")

    fast_config = get_config("fast", max_new_tokens=64)
    print(f"快速配置: {fast_config}")
