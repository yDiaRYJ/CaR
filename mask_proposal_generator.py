from generate_cam import generate_cam, zeroshot_classifier, reshape_transform
from generate_pseudo_mask import crf
from generate_heatmap import generate_heatmap
from pytorch_grad_cam import GradCAM
import torch
import clip
import os
from clip_text import class_names, new_class_names_coco, BACKGROUND_CATEGORY_COCO


if __name__ == '__main__':
    # 参数设置
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 选择设备
    model_path = 'resources/models/ViT-B-16.pt'
    image_path = 'resources/input/image/person with clothes,people,human.jpg'
    cam_out_dir = 'resources/output/cam'
    mean_bgr = (104.008, 116.669, 122.675)
    cam_path = cam_out_dir + '/' + image_path.split("/")[-1].split(".")[0] + '.npy'
    pseudo_mask_save_path = "resources/output/pseudo_mask"
    heatmap_save_dir = "resources/output/heatmap"
    # 加载CLIP模型
    model, _ = clip.load(model_path, device=device)
    # 获取背景文本特征
    bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY_COCO, ['a clean origami {}.'], model)
    # 创建GradCAM对象
    target_layers = [model.visual.transformer.resblocks[-1].ln_1]  # 设置目标层
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    # 创建输出目录
    if not os.path.exists(cam_out_dir):
        os.makedirs(cam_out_dir)
    if not os.path.exists(pseudo_mask_save_path):
        os.makedirs(pseudo_mask_save_path)
    # 定义标签
    label_list = ['person with clothes, people, human']
    # process
    generate_cam(img_path=image_path, cam_out_dir=cam_out_dir, model=model, label_list=label_list, bg_text_features=bg_text_features, cam=cam)
    crf(image_path=image_path, cam_path=cam_path, pseudo_mask_save_path=pseudo_mask_save_path)
    generate_heatmap(image_path=image_path, cam_path=cam_path,heatmap_save_dir=heatmap_save_dir)