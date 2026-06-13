# LAIGCD Backend API

LAIGCD 检测服务的后端 API，提供图片上传和 AI 生成内容检测功能。

## 功能

- 接收上传的图片
- 返回检测结果（Real/Fake）和置信度
- 返回空域热力图与原图的叠加图（第一阶段）
- 返回频域热力图与原图的叠加图（第二阶段）

## 安装

```bash
cd backend
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

服务将在 `http://localhost:8000` 启动。

## API 端点

### POST /api/detect

上传图片进行检测。

**请求:**
- Content-Type: `multipart/form-data`
- 参数:
  - `image`: 图片文件（必填）
  - `threshold`: 检测阈值（可选，默认使用 `0.68`）

**响应:**
```json
{
  "prediction": "Fake",
  "confidence": 0.85,
  "fake_probability": 0.85,
  "threshold": 0.68,
  "spatial_overlay": "base64编码的PNG图像",
  "frequency_overlay": "base64编码的PNG图像"
}
```

### GET /api/health

健康检查。

**响应:**
```json
{
  "status": "healthy"
}
```

### GET /api/info

获取模型信息。

**响应:**
```json
{
  "checkpoint_path": "checkpoints/full_run/best_model.pth",
  "device": "cuda",
  "img_size": 224,
  "default_threshold": 0.68,
  "config": {...}
}
```

## 使用示例

### Python

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# 上传图片
url = "https://frp-cat.com:16240/api/detect"
with open("test.jpg", "rb") as f:
    response = requests.post(url, files={"image": f})

result = response.json()

# 保存叠加图
if "spatial_overlay" in result:
    img_data = base64.b64decode(result["spatial_overlay"])
    img = Image.open(BytesIO(img_data))
    img.save("spatial_overlay.png")

if "frequency_overlay" in result:
    img_data = base64.b64decode(result["frequency_overlay"])
    img = Image.open(BytesIO(img_data))
    img.save("frequency_overlay.png")

print(f"预测: {result['prediction']}")
print(f"置信度: {result['confidence']:.2%}")
```

### cURL

```bash
curl -X POST "https://frp-cat.com:16240/api/detect" \
  -F "image=@test.jpg" \
  -o result.json
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('https://frp-cat.com:16240/api/detect', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('预测:', data.prediction);
  console.log('置信度:', data.confidence);
  
  // 显示叠加图
  if (data.spatial_overlay) {
    const img = document.getElementById('spatial-overlay');
    img.src = 'data:image/png;base64,' + data.spatial_overlay;
  }
  if (data.frequency_overlay) {
    const img = document.getElementById('frequency-overlay');
    img.src = 'data:image/png;base64,' + data.frequency_overlay;
  }
});
```

## 配置

修改 `detector.py` 中的默认参数：

- `checkpoint_path`: 模型检查点路径
- `device`: 运行设备（`cuda` 或 `cpu`）
- `img_size`: 输入图像大小（默认 224）

## 注意事项

1. 模型文件需要预先训练并放置在 `checkpoints/` 目录
2. 首次运行会加载模型，可能需要几秒钟
3. `frequency_overlay` 仅在使用频域特征的模型中返回
