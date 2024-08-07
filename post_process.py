import json

import torch
import os
import numpy as np
import torch.nn.functional as F
import joblib
import multiprocessing
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import cv2
from PIL import Image
import argparse
from generate_heatmap import generate_heatmap
from pycocotools import mask as mask_utils

mean_bgr = (104.008, 116.669, 122.675)

class DenseCRF(object):
    """
    DenseCRF类，用于进行CRF（条件随机场）后处理
    """

    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        # 初始化CRF的参数
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        """
        执行CRF后处理

        :param image: 输入的图像
        :param probmap: 概率图
        :return: 后处理后的结果
        """
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)  # 从softmax概率图生成一元能量
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)  # 初始化DenseCRF
        d.setUnaryEnergy(U)  # 设置一元能量
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)  # 添加对数高斯项
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )  # 添加对数双边项

        Q = d.inference(self.iter_max)  # 进行推理
        Q = np.array(Q).reshape((C, H, W))

        return Q  # 返回结果


def makedirs(dirs):
    """
    创建目录，如果目录不存在的话

    :param dirs: 目录路径
    """
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def _fast_hist(label_true, label_pred, n_class):
    """
    计算混淆矩阵

    :param label_true: 真实标签
    :param label_pred: 预测标签
    :param n_class: 类别数量
    :return: 混淆矩阵
    """
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    """
    计算评估指标

    :param label_trues: 真实标签列表
    :param label_preds: 预测标签列表
    :param n_class: 类别数量
    :return: 评估指标的字典
    """
    hist = np.zeros((n_class, n_class))  # 初始化混淆矩阵
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)  # 累加混淆矩阵
    acc = np.diag(hist).sum() / hist.sum()  # 像素准确率
    acc_cls = np.diag(hist) / hist.sum(axis=1)  # 类别准确率
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))  # 交并比
    valid = hist.sum(axis=1) > 0  # 有效类别
    mean_iu = np.nanmean(iu[valid])  # 平均交并比
    freq = hist.sum(axis=1) / hist.sum()  # 类别频率
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()  # 频权交并比
    cls_iu = dict(zip(range(n_class), iu))  # 类别交并比

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }



def crf_coco14(image_path, cam_dic, pseudo_mask_save_path='', save=False):
    """
    对生成的cam进行CRF后处理，并生成分割图

    :param image_path: 图片路径
    :param cam_dic: cam字典
    :param pseudo_mask_save_path: 掩码保存文件路径
    :param save: 是否保存掩码文件
    """
    # 配置
    # torch.set_grad_enabled(False)

    # CRF后处理器
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    # process
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)  # 读取图像
    image -= mean_bgr  # 均值减法
    image = image.transpose(2, 0, 1)  # HWC -> CHW

    cams = cam_dic['attn_highres']
    bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 1)  # 计算背景分数
    cams = np.concatenate((bg_score, cams), axis=0)
    prob = cams

    image = image.astype(np.uint8).transpose(1, 2, 0)  # 转换图像格式
    prob = postprocessor(image, prob)  # 执行CRF后处理

    label = np.argmax(prob, axis=0)  # 预测标签
    confidence = np.max(prob, axis=0)
    pseudo_mask = label.astype(np.uint8)
    if save:
        cv2.imwrite(pseudo_mask_save_path, pseudo_mask)  # 保存伪标签
    return pseudo_mask, confidence


def post_process_coco14(image_path, cam_dics, label_list, pseudo_mask_save_path='', save=False):
    mask_list = []
    confidence_list = []
    from clip_text import new_class_names_coco
    for cam_dic, label in zip(cam_dics, label_list):
        label_idx = new_class_names_coco.index(label) + 1
        mask, confidence = crf_coco14(image_path, cam_dic, save=save)
        mask[mask == 1] = label_idx
        mask_list.append(mask)
        confidence_list.append(confidence)

    # 创建一个空的目标数组
    final_mask = np.zeros_like(mask_list[0])

    # 创建一个空的 target_confidence 数组
    target_confidence = np.zeros_like(mask_list[0], dtype=np.float32)

    # 遍历 mask_list,将非零元素填入目标数组
    for i, mask in enumerate(mask_list):
        final_mask[mask != 0] = mask[mask != 0]
        target_confidence[mask != 0] = confidence_list[i][mask != 0]

    # 遍历 target_array,如果当前位置不为 0,则比较 target_confidence 和 mask 对应位置的 confidence
    for i in range(final_mask.shape[0]):
        for j in range(final_mask.shape[1]):
            if final_mask[i, j] != 0:
                for k, mask in enumerate(mask_list):
                    if mask[i, j] != 0:
                        if confidence_list[k][i, j] > target_confidence[i, j]:
                            final_mask[i, j] = mask[i, j]
                            target_confidence[i, j] = confidence_list[k][i, j]

    # target_array[target_confidence < 0.95] = 255  # 低置信度区域设为255

    return final_mask


def crf(image_path, cam_dic, pseudo_mask_save_path='', save=False):
    """
    对生成的cam进行CRF后处理，并生成分割图

    :param image_path: 图片路径
    :param cam_dic: cam字典
    :param pseudo_mask_save_path: 掩码保存文件路径
    :param save: 是否保存掩码文件
    """
    # 配置
    # torch.set_grad_enabled(False)

    # CRF后处理器
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    # process
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)  # 读取图像
    image -= mean_bgr  # 均值减法
    image = image.transpose(2, 0, 1)  # HWC -> CHW

    cams = cam_dic['attn_highres']
    bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 1)  # 计算背景分数
    cams = np.concatenate((bg_score, cams), axis=0)
    prob = cams

    image = image.astype(np.uint8).transpose(1, 2, 0)  # 转换图像格式
    prob = postprocessor(image, prob)  # 执行CRF后处理

    keys = [0, 56, 58, 60, 62, 68, 72, 73, 74, 75]
    keys = torch.tensor(keys)
    keys = np.pad(keys + 1, (1, 0), mode='constant')

    label = np.argmax(prob, axis=0)  # 预测标签
    label = keys[label]
    confidence = np.max(prob, axis=0)
    label[confidence < 0.95] = 255  # 低置信度区域设为255
    pseudo_mask = label.astype(np.uint8)
    if save:
        cv2.imwrite(pseudo_mask_save_path, pseudo_mask)  # 保存伪标签
    return pseudo_mask


def post_process(image_path, cam_dics, pseudo_mask_save_path='', save=False):
    final_cam = None
    for cam_dic in cam_dics:
        cam = cam_dic['attn_highres']
        if final_cam is None:
            final_cam = cam.copy()
        else:
            final_cam = np.maximum(final_cam, cam)
    final_cam_dic = {
        'attn_highres': final_cam
    }
    final_heatmap = generate_heatmap(image_path=image_path, cam_dic=final_cam_dic, save=save)
    final_mask = crf(image_path=image_path, cam_dic=final_cam_dic, save=save)
    return final_heatmap, final_mask


def binary_mask_to_rle(binary_mask):
    """
    将二值分割掩码转换为rle格式。
    """
    # contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # segmentation = []
    # for contour in contours:
    #     if contour.size >= 6:  # 至少需要3个点来表示一个多边形
    #         segmentation.append(contour.flatten().tolist())
    # return segmentation
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle['counts'] = rle['counts'].decode('utf-8')  # RLE编码结果是字节，需要解码为字符串
    return rle


def post_process_lvis(image_path, label_list, cam_dics, pseudo_mask_save_path='', save=False):
    results = []
    from lvis_text import class_names_lvis
    for label, cam_dic in zip(label_list, cam_dics):
        mask = crf(image_path=image_path, cam_dic=cam_dic, save=save)
        mask[mask == 255] = 1
        rle = binary_mask_to_rle(mask)
        image_id = os.path.basename(image_path).split(".")[0]
        category_id = class_names_lvis.index(label) + 1
        result = {
            "image_id": int(image_id),
            "category_id": category_id,
            "segmentation": rle,
            "area": mask_utils.area(rle).item(),
            "bbox": mask_utils.toBbox(rle).tolist(),
            "score": 1.0
        }
        results.append(result)

    return results


if __name__ == "__main__":
    pseudo_mask_save_path = "resources/output/pseudo_mask"
    image_path = "resources/input/image/1.jpg"
    cam_path = "resources/output/cam/1.npy"

    if not os.path.exists(pseudo_mask_save_path):
        os.makedirs(pseudo_mask_save_path)

    crf(image_path=image_path, cam_path=cam_path, pseudo_mask_save_path=pseudo_mask_save_path)