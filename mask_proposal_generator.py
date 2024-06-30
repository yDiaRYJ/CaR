from generate_cam import generate_cam, zeroshot_classifier, reshape_transform, image_preprocess
from generate_pseudo_mask import crf
from generate_heatmap import generate_heatmap
from visual_prompting import blur_background
from pytorch_grad_cam import GradCAM
import torch
import clip
import os
from clip_text import BACKGROUND_CATEGORY_COCO


if __name__ == '__main__':
    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 选择设备
    model_path = 'resources/models/ViT-B-16.pt'
    model, _ = clip.load(model_path, device=device)
    # 获取背景文本特征
    bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY_COCO, ['a clean origami {}.'], model).to(device)
    # 创建GradCAM对象
    target_layers = [model.visual.transformer.resblocks[-1].ln_1]  # 设置目标层
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    mean_bgr = (104.008, 116.669, 122.675)

    # 输入输出设置
    # 图片路径
    image_path = 'resources/input/image/test1.jpg'
    image_name = image_path.split("/")[-1].split(".")[0]
    image_info = image_preprocess(img_path=image_path, model=model)
    # 存储路径
    cam_out_dir = f'resources/output/cam/{image_name}'
    pseudo_mask_save_dir = f"resources/output/pseudo_mask/{image_name}"
    heatmap_save_dir = f"resources/output/heatmap/{image_name}"
    visual_prompt_save_dir = f"resources/output/visual_prompt/{image_name}"
    # 创建输出目录
    if not os.path.exists(cam_out_dir):
        os.makedirs(cam_out_dir)
    if not os.path.exists(pseudo_mask_save_dir):
        os.makedirs(pseudo_mask_save_dir)
    if not os.path.exists(heatmap_save_dir):
        os.makedirs(heatmap_save_dir)
    if not os.path.exists(visual_prompt_save_dir):
        os.makedirs(visual_prompt_save_dir)
    # 定义标签
    label_list = ['person with clothes', 'people', 'human', 'phone', 'face']
    # 生成所有cam文件
    generate_cam(image_info=image_info, cam_out_dir=cam_out_dir, model=model, label_list=label_list,
                 bg_text_features=bg_text_features, cam=cam)
    # 生成所有掩码文件
    for label in label_list:
        cam_out_path = f'{cam_out_dir}/{label}.npy'
        pseudo_mask_save_path = f"{pseudo_mask_save_dir}/{label}.png"
        heatmap_save_path = f"{heatmap_save_dir}/{label}.png"
        visual_prompt_save_path = f"{visual_prompt_save_dir}/{label}.png"
        # process
        crf(image_path=image_path, cam_path=cam_out_path, pseudo_mask_save_path=pseudo_mask_save_path)
        generate_heatmap(image_path=image_path, cam_path=cam_out_path,heatmap_save_path=heatmap_save_path)
        blur_background(image_path=image_path, cam_path=cam_out_path, threshold=0.5, output_path=visual_prompt_save_path)
