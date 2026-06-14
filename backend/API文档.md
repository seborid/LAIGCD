# 详细接口说明

服务地址：https://frp-cat.com:16240/

### GET `/`

返回服务基础信息。

#### 响应示例

```json
{
  "message": "LAIGCD Detection API",
  "version": "1.0.0",
  "endpoints": {
    "detect": "/api/detect",
    "health": "/api/health"
  }
}
```

#### 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `message` | string | 服务名称 |
| `version` | string | 服务版本 |
| `endpoints.detect` | string | 检测接口路径 |
| `endpoints.health` | string | 健康检查接口路径 |

---

### GET `/api/health`

用于健康检查，不触发模型加载。

#### 响应示例

```json
{
  "status": "healthy"
}
```

#### 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | string | 固定为 `healthy` |

---

### 3 GET `/api/info`

返回当前模型配置与运行信息。首次访问时会触发模型加载。

#### 响应示例

```json
{
  "checkpoint_path": "checkpoints/full_run/best_model.pth",
  "device": "cuda",
  "img_size": 224,
  "default_threshold": 0.68,
  "config": {
    "clip_model": "ViT-B/32",
    "num_prototypes": 16,
    "use_freq": true,
    "freq_type": "srm",
    "dropout": 0.1
  }
}
```

#### 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `checkpoint_path` | string | 模型检查点路径 |
| `device` | string | 当前推理设备，通常为 `cuda` 或 `cpu` |
| `img_size` | integer | 输入图像尺寸 |
| `default_threshold` | number | 默认推理阈值 |
| `config` | object | 从检查点目录下 `config.json` 读取的模型配置；若不存在则使用代码中的默认配置 |

---

### 4 POST `/api/detect`

上传单张图片并执行 AI 生成内容检测。

#### 请求格式

- `Content-Type: multipart/form-data`

#### 表单参数

| 参数名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `image` | file | 是 | 待检测图片 |
| `threshold` | float | 否 | 自定义检测阈值；不传时默认使用 `0.68` |
| `explain` | boolean | 否 | 是否调用第二阶段 FakeVLM 生成自然语言理由，默认 `true`。传 `false` 可跳过 7B 模型推理以加快响应 |

#### 请求示例

```bash
curl -X POST "http://localhost:8000/api/detect" \
  -F "image=@test.jpg" \
  -F "threshold=0.68"
```

跳过第二阶段 FakeVLM 推理（仅返回分类结果与热力图，响应更快）：

```bash
curl -X POST "http://localhost:8000/api/detect" \
  -F "image=@test.jpg" \
  -F "explain=false"
```

#### 成功响应示例

```json
{
  "prediction": "Fake",
  "confidence": 0.9132,
  "fake_probability": 0.9132,
  "threshold": 0.68,
  "spatial_overlay": "iVBORw0KGgoAAAANSUhEUgAA...",
  "frequency_overlay": "iVBORw0KGgoAAAANSUhEUgAA...",
  "reasoning": "This image is fake. First, the lighting on the face is inconsistent..."
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `prediction` | string | 检测结果，取值为 `Real` 或 `Fake` |
| `confidence` | number | 当前预测标签的置信度，范围通常为 `0.0 ~ 1.0` |
| `fake_probability` | number | 模型输出的伪造概率，范围通常为 `0.0 ~ 1.0` |
| `threshold` | number | 本次请求使用或传入的阈值 |
| `spatial_overlay` | string | 空域热力图叠加图，base64 编码 PNG |
| `frequency_overlay` | string | 频域热力图叠加图，base64 编码 PNG；只有模型支持频域分支时才返回 |
| `reasoning` | string \| null | 第二阶段 FakeVLM 给出的真伪理由 / 伪影解释（英文）。当 `explain=false` 或模型调用失败时为 `null` |

#### 业务说明

- 图片在服务端会被转换为 `RGB`。
- 返回的热力图不是原始数组，而是已经叠加到原图上的 PNG 图像。
- `spatial_overlay` 一定返回。
- `frequency_overlay` 可能不存在，前端需要做空值兼容。
- `reasoning` 为两阶段流水线的第二阶段输出：第一阶段（CLIP 分类器）给出 `prediction` 后，FakeVLM（LLaVA-1.5-7B）被告知该判定并生成理由。开启 `explain` 时单次请求会额外占用约 7-8GB 显存、增加约 15-25 秒延迟；首次请求需懒加载 7B 模型（约 20-40 秒，仅一次）。FakeVLM 调用失败会降级为 `reasoning: null`，不影响分类结果与热力图。
- 说明：在 12GB 显存下 FakeVLM 以 8-bit 量化加载，模型对图像真伪的解释质量随图像而异——对明确的伪造图像（与 `Fake` 判定一致时）通常能给出 3 条具体理由；对真实图像或低置信情形可能输出较为笼统的描述，甚至与 `Real` 判定相悖（模型整体偏 fake）。`reasoning` 仅供可解释性参考，真伪判定以第一阶段的 `prediction`/`confidence` 为准。

#### Python 调用示例

```python
import base64
from io import BytesIO

import requests
from PIL import Image

url = "https://frp-cat.com:16240/api/detect"

with open("test.jpg", "rb") as f:
    response = requests.post(
        url,
        files={"image": f},
        data={"threshold": "0.68"},
        timeout=60,
    )

result = response.json()

print(result["prediction"])
print(result["confidence"])
print(result["fake_probability"])

spatial = base64.b64decode(result["spatial_overlay"])
Image.open(BytesIO(spatial)).save("spatial_overlay.png")

if "frequency_overlay" in result:
    frequency = base64.b64decode(result["frequency_overlay"])
    Image.open(BytesIO(frequency)).save("frequency_overlay.png")
```

#### JavaScript 调用示例

```javascript
const formData = new FormData();
formData.append("image", fileInput.files[0]);
formData.append("threshold", "0.68");

const response = await fetch("https://frp-cat.com:16240/api/detect", {
  method: "POST",
  body: formData
});

const data = await response.json();

console.log(data.prediction);
console.log(data.fake_probability);

document.getElementById("spatial-overlay").src =
  `data:image/png;base64,${data.spatial_overlay}`;

if (data.frequency_overlay) {
  document.getElementById("frequency-overlay").src =
    `data:image/png;base64,${data.frequency_overlay}`;
}
```