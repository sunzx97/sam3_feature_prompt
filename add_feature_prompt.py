import os
import torch
from PIL import Image
import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox

# 设置设备
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

# 特征保存路径
feature_save_path = "config/feature_prompt.pkl"

# 如果文件存在，先删除
if os.path.exists(feature_save_path):
    os.remove(feature_save_path)
    print(f"已删除旧的特征文件: {feature_save_path}")

# 初始化处理器
processor = Sam3Processor(model, confidence_threshold=0.5)

# 处理每个相机配置
for camera_config in cameraList:
    camera_code = camera_config["cameraCode"]
    image_path = camera_config["imagePath"]
    part_box = camera_config["part"]  # [x, y, w, h]
    label = camera_config["label"]

    print(f"\n处理相机 {camera_code}: {image_path}")

    # 加载图像
    image = Image.open(image_path)
    width, height = image.size

    # 设置图像
    inference_state = processor.set_image(image)

    # 重置提示
    processor.reset_all_prompts(inference_state)

    # 转换box格式：xywh -> cxcywh
    box_input_xywh = torch.tensor(part_box).view(-1, 4)
    box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)

    # 归一化box
    norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
    print(f"归一化box输入: {norm_box_cxcywh}")

    # 添加几何提示并保存特征
    inference_state = processor.add_geometric_prompt(
        state=inference_state,
        box=norm_box_cxcywh,
        label=label,
        cache_path=feature_save_path,
        camera_code=camera_code,
    )

    print(f"相机 {camera_code} 的特征已保存到: {feature_save_path}")

print("\n所有特征提取完成！")
