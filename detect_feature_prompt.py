import os
import torch
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# 导入配置
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import cameraList

# 模型路径
ckpt = r"/home/sun/.cache/modelscope/hub/models/facebook/sam3/sam3.pt"
model = build_sam3_image_model(
    checkpoint_path=ckpt,
    load_from_HF=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
)

# 特征文件路径
feature_load_path = "config/feature_prompt.pkl"

# 加载特征字典
if not os.path.exists(feature_load_path):
    raise FileNotFoundError(f"特征文件不存在: {feature_load_path}，请先运行 add_feature_prompt.py")

with open(feature_load_path, "rb") as f:
    feature_dict = pickle.load(f)


print(f"已加载特征文件")
print(f"可用的相机代码: {list(feature_dict.keys())}")

# 指定要推理的相机代码
target_camera_code = "001"  # 修改这里为你想要推理的相机代码

if target_camera_code not in feature_dict:
    raise ValueError(f"相机代码 {target_camera_code} 不在特征文件中")

# 获取该相机对应的特征列表
camera_feature_list = feature_dict[target_camera_code]
print(f"\n相机 {target_camera_code} 共有 {len(camera_feature_list)} 个特征")


image_path = f"/home/sun/data/coding/object_detection_based_on_context_learning/SAM_based/VOC2012_train_val/JPEGImages/2008_000003.jpg"

image = Image.open(image_path)
width, height = image.size
processor = Sam3Processor(model, confidence_threshold=0.5)
inference_state = processor.set_image(image)

processor.reset_all_prompts(inference_state)
inference_state = processor.set_text_prompt(state=inference_state, prompt="person")

img0 = Image.open(image_path)
plot_results(img0, inference_state)

plt.imshow(img0)
plt.axis("off")  # Hide the axis
plt.show()

for feature_prompt_item in camera_feature_list:
    feature_prompt = feature_prompt_item["prompt"].to(model.device)
    feature_prompt_mask=  feature_prompt_item["prompt_mask"].to(model.device)
    inference_state = processor.add_geometric_prompt(
        state=inference_state,
        load_from_cache=True,
        feature_prompt=feature_prompt,
        feature_prompt_mask=feature_prompt_mask
    )

img0 = Image.open(image_path)
plot_results(img0, inference_state)
plt.axis("off")  # Hide the axis
plt.show()