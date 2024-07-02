from generate_cam import generate_cam, zeroshot_classifier, reshape_transform, image_preprocess
from generate_heatmap import generate_heatmap
from visual_prompting import blur_background
from clip_es.pytorch_grad_cam import GradCAM
import torch
from clip_es import clip
import os
from clip_text import BACKGROUND_CATEGORY_COCO


def mask_proposal_generator(model, cam, bg_text_features, image_path, label_list, output_path="resources/output", save=False, eta=0.5):
    # 图片信息
    image_name = image_path.split("/")[-1].split(".")[0]
    image_info = image_preprocess(img_path=image_path, model=model)
    # 存储路径
    cam_out_dir = f'{output_path}/cam/{image_name}'
    pseudo_mask_save_dir = f"{output_path}/pseudo_mask/{image_name}"
    heatmap_save_dir = f"{output_path}/heatmap/{image_name}"
    visual_prompt_save_dir = f"{output_path}/visual_prompt/{image_name}"
    # 创建输出目录
    if not os.path.exists(cam_out_dir):
        os.makedirs(cam_out_dir)
    if not os.path.exists(pseudo_mask_save_dir):
        os.makedirs(pseudo_mask_save_dir)
    if not os.path.exists(heatmap_save_dir):
        os.makedirs(heatmap_save_dir)
    if not os.path.exists(visual_prompt_save_dir):
        os.makedirs(visual_prompt_save_dir)
    # 生成cam
    cam_dic_list = generate_cam(image_info=image_info, cam_out_dir=cam_out_dir, model=model, label_list=label_list,
                 bg_text_features=bg_text_features, cam=cam, save=save)
    visual_prompt_list = []
    # 生成掩码
    for index, label in enumerate(label_list):
        cam_dic = cam_dic_list[index]
        pseudo_mask_save_path = f"{pseudo_mask_save_dir}/{label}.png"
        heatmap_save_path = f"{heatmap_save_dir}/{label}.png"
        visual_prompt_save_path = f"{visual_prompt_save_dir}/{label}.png"
        # process
        heatmap = generate_heatmap(image_path=image_path, cam_dic=cam_dic,heatmap_save_path=heatmap_save_path, save=save)
        visual_prompt = blur_background(image_path=image_path, cam_dic=cam_dic, threshold=eta, output_path=visual_prompt_save_path, save=save)
        visual_prompt_list.append(visual_prompt)
    return cam_dic_list, visual_prompt_list


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

    # 输入输出设置
    # 图片路径
    image_path = 'resources/input/image/test1.jpg'
    # 定义标签
    label_list = ['person with clothes,people,human', 'cell phone', 'face', 'building', 'dog']
    # 输出路径
    output_path = "resources/output"
    mask_proposal_generator(model=model, cam=cam, bg_text_features=bg_text_features, image_path=image_path, label_list=label_list, output_path=output_path)

