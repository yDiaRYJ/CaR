import os

import numpy as np
from PIL import Image


def _fast_hist(label_true, label_pred, n_class):
    # 创建一个布尔掩码,过滤掉无效的标签
    mask = (label_true >= 0) & (label_true < n_class)
    # 使用 np.bincount 统计每个类别的预测次数,并将结果拉平成一维数组
    # minlength=n_class ** 2 确保输出数组长度为 n_class^2
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    )
    return hist.reshape(n_class, n_class)


def scores(label_trues, label_preds, n_class):
    # 初始化一个 n_class x n_class 的直方图矩阵
    hist = np.zeros((n_class, n_class))

    # 遍历所有的真实标签和预测标签,累加它们的直方图
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    # 计算像素级准确率
    acc = np.diag(hist).sum() / hist.sum()

    # 计算平均准确率
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    # 计算 IoU (Intersection over Union)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    # 找到有效的类别(即标签总数大于 0 的类别)
    valid = hist.sum(axis=1) > 0

    # 计算平均 IoU
    mean_iu = np.nanmean(iu[valid])

    # 计算频率加权 IoU
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    # 将每个类别的 IoU 值存储到字典中
    cls_iu = dict(zip(range(n_class), iu))

    # 返回评估指标
    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


def eval(label_predict_dir, label_true_dir, n_class):
    label_predicts = []
    label_trues = []
    # 遍历文件夹下所有文件
    for label_predict_name in os.listdir(label_predict_dir):
        # 检查文件是否为 PNG 格式
        if label_predict_name.endswith('.png'):
            # 获取完整的文件路径
            label_predict_path = os.path.join(label_predict_dir, label_predict_name)
            label_true_name = label_predict_name.split("_")[-1]
            label_true_path = os.path.join(label_true_dir, label_true_name)
            # 获取图片文件
            label_predict = np.asarray(Image.open(label_predict_path), dtype=np.uint8)
            label_true = np.asarray(Image.open(label_true_path), dtype=np.uint8)
            # 加入列表
            label_predicts.append(label_predict)
            label_trues.append(label_true)
    score = scores(label_trues, label_predicts, n_class)
    return score

if __name__ == '__main__':
    label_predict_dir = "D:\et\program\code\python\zju\dataset\coco2014\MyResult1\mask"
    label_true_dir = "D:\et\program\code\python\zju\dataset\coco2014\SegmentationClass"
    is_coco = True
    n_class = 21 if not is_coco else 81
    print(eval(label_predict_dir, label_true_dir, n_class))
