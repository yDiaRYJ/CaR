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


def crf(image_path, cam_path, pseudo_mask_save_path):
    """
    对生成的cam进行CRF后处理，并生成分割图

    :param image_path: 图片路径
    :param cam_path: cam文件路径
    :param pseudo_mask_save_path: 掩码保存文件夹路径
    """
    # 配置
    torch.set_grad_enabled(False)

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
    image_name = image_path.split("/")[-1].split(".")[0]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)  # 读取图像
    image -= mean_bgr  # 均值减法
    image = image.transpose(2, 0, 1)  # HWC -> CHW

    cam_dict = np.load(cam_path, allow_pickle=True).item()  # 读取cam字典
    cams = cam_dict['attn_highres']
    bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 1)  # 计算背景分数
    cams = np.concatenate((bg_score, cams), axis=0)
    prob = cams

    image = image.astype(np.uint8).transpose(1, 2, 0)  # 转换图像格式
    prob = postprocessor(image, prob)  # 执行CRF后处理

    label = np.argmax(prob, axis=0)  # 预测标签
    confidence = np.max(prob, axis=0)
    label[confidence < 0.95] = 255  # 低置信度区域设为255
    cv2.imwrite(os.path.join(pseudo_mask_save_path, image_name + '.png'), label.astype(np.uint8))  # 保存伪标签

    return 0


if __name__ == "__main__":
    pseudo_mask_save_path = "resources/output/pseudo_mask"
    image_path = "resources/input/image/1.jpg"
    cam_path = "resources/output/cam/1.npy"

    if not os.path.exists(pseudo_mask_save_path):
        os.makedirs(pseudo_mask_save_path)

    crf(image_path=image_path, cam_path=cam_path, pseudo_mask_save_path=pseudo_mask_save_path)