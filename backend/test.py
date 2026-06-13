import requests
import base64
from PIL import Image
from io import BytesIO

# 上传图片
url = "https://frp-cat.com:16240/api/detect"
with open("1.jpg", "rb") as f:
    response = requests.post(url, files={"image": f}, verify=False)

result = response.json()
print(result)
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