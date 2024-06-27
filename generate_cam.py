# -*- coding:UTF-8 -*-
# 导入必要的库
from pytorch_grad_cam import GradCAM
import torch
import clip
from PIL import Image
import numpy as np
import cv2
import os
from pytorch_grad_cam.utils.image import scale_cam_image
from utils import scoremap2bbox
from clip_text import class_names, new_class_names_coco, BACKGROUND_CATEGORY_COCO
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import warnings
warnings.filterwarnings("ignore")

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

device = "cuda" if torch.cuda.is_available() else "cpu"  # 选择设备


def reshape_transform(tensor, height=28, width=28):
    """
    将Transformer网络的输出重塑为CNN网络的形状。

    :param tensor: 输入张量
    :param height: 重塑后的高度
    :param width: 重塑后的宽度
    :return: 重塑后的张量
    """
    tensor = tensor.permute(1, 0, 2)  # 调整维度顺序
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))  # 重塑为指定高度和宽度
    result = result.transpose(2, 3).transpose(1, 2)  # 将通道维度放到第一个维度
    return result


def zeroshot_classifier(classnames, templates, model):
    """
    基于零样本学习的分类器，使用CLIP模型进行文本编码。

    :param classnames: 类别名称列表
    :param templates: 文本模板列表
    :param model: CLIP模型
    :return: 类别特征向量
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # 格式化文本模板
            texts = clip.tokenize(texts).to(device)  # 将文本标记化
            class_embeddings = model.encode_text(texts)  # 使用CLIP模型编码文本
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # 归一化
            class_embedding = class_embeddings.mean(dim=0)  # 取平均
            class_embedding /= class_embedding.norm()  # 再次归一化
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)  # 堆叠为矩阵
    return zeroshot_weights.t()


class ClipOutputTarget:
    """
    用于提取模型输出中特定类别的值的目标类。
    """
    def __init__(self, category):
        self.category = category  # 初始化类别索引

    def __call__(self, model_output):
        """
        根据类别索引返回模型输出的特定值。

        :param model_output: 模型的输出，可能是一维或二维的张量
        :return: 返回模型输出中对应类别索引的值
        """
        if len(model_output.shape) == 1:
            return model_output[self.category]  # 如果输出是一维的，直接返回对应类别的值
        return model_output[:, self.category]  # 如果输出是二维的，返回对应类别的值


def _convert_image_to_rgb(image):
    """
    将图像转换为RGB格式。

    :param image: 输入的图像
    :return: 转换为RGB格式的图像
    """
    return image.convert("RGB")


def _transform_resize(h, w):
    """
    构建一个用于调整图像大小和标准化的变换函数。

    :param h: 调整后的图像高度
    :param w: 调整后的图像宽度
    :return: 组合的图像变换函数
    """
    return Compose([
        Resize((h, w), interpolation=BICUBIC),  # 调整图像大小
        _convert_image_to_rgb,  # 转换为RGB格式
        ToTensor(),  # 转换为张量
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # 标准化
    ])


def img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0], patch_size=16):
    """
    生成多尺度和翻转后的图像。

    :param img_path: 图像路径
    :param ori_height: 原始图像高度
    :param ori_width: 原始图像宽度
    :param scales: 用于调整图像大小的比例列表
    :param patch_size: 补丁大小，用于调整尺寸
    :return: 包含多尺度和翻转后的图像列表
    """
    all_imgs = []  # 存储所有处理后的图像
    for scale in scales:
        preprocess = _transform_resize(
            int(np.ceil(scale * int(ori_height) / patch_size) * patch_size),  # 调整后的高度
            int(np.ceil(scale * int(ori_width) / patch_size) * patch_size)  # 调整后的宽度
        )
        image = preprocess(Image.open(img_path))  # 打开并处理图像
        image_ori = image  # 原始图像
        image_flip = torch.flip(image, [-1])  # 水平翻转图像
        all_imgs.append(image_ori)  # 添加原始图像到列表
        all_imgs.append(image_flip)  # 添加翻转图像到列表
    return all_imgs  # 返回处理后的图像列表


def generate_cam(img_path, cam_out_dir, model, label_list, bg_text_features, cam):
    """
    检测单张图片，执行CAM生成任务。

    :param img_path: 图片路径
    :param cam_out_dir: 输出cam文件路径
    :param model: CLIP模型
    :param label_list: 该图片的前景文本列表
    :param bg_text_features: 背景文本特征
    :param cam: CAM生成器
    """
    model = model.to(device)  # 将模型移动到对应设备上
    bg_text_features = bg_text_features.to(device)  # 将背景文本特征移动到对应设备上
    fg_text_features = zeroshot_classifier(label_list, ['a clean origami {}.'], model)  # 获取前景文本特征
    fg_text_features = fg_text_features.to(device)  # 将前景文本特征移动到对应设备上

    ori_image = Image.open(img_path)  # 打开图像
    ori_height, ori_width = np.asarray(ori_image).shape[:2]  # 获取图像的原始高度和宽度

    ms_imgs = img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0])  # 获取多尺度图像
    ms_imgs = [ms_imgs[0]]  # 只使用第一个尺度的图像

    highres_cam_all_scales = []
    refined_cam_all_scales = []

    # 获取识别的图片
    image = ms_imgs[0]
    image = image.unsqueeze(0)  # 添加批次维度
    h, w = image.shape[-2], image.shape[-1]  # 获取图像的高度和宽度
    image = image.to(device)  # 将图像移动到对应设备上
    image_features, attn_weight_list = model.encode_image(image, h, w)  # 编码图像，获取图像特征和注意力权重

    highres_cam_to_save = []
    refined_cam_to_save = []

    bg_features_temp = bg_text_features.to(device)  # 将背景特征移动到对应设备上
    fg_features_temp = fg_text_features.to(device)  # 获取前景特征并移动到对应设备上
    text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)  # 合并前景和背景特征
    input_tensor = [image_features, text_features_temp.to(device), h, w]  # 构建输入张量

    for idx, label in enumerate(label_list):
        targets = [ClipOutputTarget(label_list.index(label))]  # 设置目标

        # 生成CAM
        grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                targets=targets,
                                                                target_size=None)
        grayscale_cam = grayscale_cam[0, :]  # 获取CAM

        grayscale_cam_highres = cv2.resize(grayscale_cam, (ori_width, ori_height))  # 调整CAM大小到原始图像尺寸
        highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))  # 保存高分辨率的CAM

        if idx == 0:
            attn_weight_list.append(attn_weight_last)
            attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # 获取注意力权重
            attn_weight = torch.stack(attn_weight, dim=0)[-8:]
            attn_weight = torch.mean(attn_weight, dim=0)
            attn_weight = attn_weight[0].cpu().detach()  # 将注意力权重移到CPU并分离
        attn_weight = attn_weight.float()  # 转换为浮点类型

        # 生成边界框
        box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.7, multi_contour_eval=True)
        aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1]))  # 创建掩码
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1  # 更新掩码

        aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])  # 重塑掩码
        aff_mat = attn_weight

        trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)  # 计算转移矩阵
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

        for _ in range(2):
            trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
            trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
        trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

        for _ in range(1):
            trans_mat = torch.matmul(trans_mat, trans_mat)

        trans_mat = trans_mat * aff_mask  # 应用掩码到转移矩阵

        cam_to_refine = torch.FloatTensor(grayscale_cam)
        cam_to_refine = cam_to_refine.view(-1, 1)  # 重塑CAM

        cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h // 16, w // 16)  # 细化CAM
        cam_refined = cam_refined.cpu().numpy().astype(np.float32)
        cam_refined_highres = scale_cam_image([cam_refined], (ori_width, ori_height))[0]  # 调整细化后的CAM大小
        refined_cam_to_save.append(torch.tensor(cam_refined_highres))  # 保存细化后的CAM

        highres_cam_all_scales.append(torch.stack(highres_cam_to_save, dim=0))  # 保存高分辨率的CAM
        refined_cam_all_scales.append(torch.stack(refined_cam_to_save, dim=0))  # 保存细化后的CAM

    highres_cam_all_scales = highres_cam_all_scales[0]
    refined_cam_all_scales = refined_cam_all_scales[0]
    attn_highres = refined_cam_all_scales.cpu().numpy().astype(np.float16)

    # 保存CAM结果到文件
    img_name = img_path.split("/")[-1]
    np.save(os.path.join(cam_out_dir, img_name.replace('jpg', 'npy')),
            { "attn_highres": attn_highres,
             })
    return attn_highres  # 返回attn_highres


if __name__ == "__main__":
    model_path = 'resources/models/ViT-B-16.pt'
    image_path = 'resources/input/image/1.jpg'
    cam_out_dir = 'resources/output/cam'
    model, _ = clip.load(model_path, device=device)  # 加载CLIP模型
    bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY_COCO, ['a clean origami {}.'], model)  # 获取背景文本特征
    if not os.path.exists(cam_out_dir):
        os.makedirs(cam_out_dir)  # 创建输出目录
    target_layers = [model.visual.transformer.resblocks[-1].ln_1]  # 设置目标层
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)  # 创建GradCAM对象
    label_list = ["tree"]
    generate_cam(img_path=image_path, cam_out_dir=cam_out_dir, model=model, label_list=label_list, bg_text_features=bg_text_features, cam=cam)