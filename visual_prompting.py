import cv2
import numpy as np


def blur_background(image_path, cam_dic, threshold, output_path, save=False):
    """
    对图像背景进行模糊处理

    :param image_path: 原始图像路径
    :param cam_dic: CAM字典
    :param threshold: CAM二值化阈值
    :param output_path: 保存结果的路径
    :param save: 是否保存文件
    """
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)

    # 读取CAM
    cam = cam_dic['attn_highres']
    cam = cam.squeeze()

    # 二值化CAM
    binary_cam = np.where(cam > threshold, 1, 0).astype(np.uint8)

    # 将原始图像从BGR转换为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 对图像进行模糊处理
    # 核大小参数 (ksize_x, ksize_y)，核大小越大，模糊效果越明显。
    blurred_image = cv2.GaussianBlur(image_rgb, (29, 29), 0)

    # 创建掩膜
    mask = np.repeat(binary_cam[:, :, np.newaxis], 3, axis=2)

    # 应用掩膜
    result = np.where(mask == 1, image_rgb, blurred_image)

    # 保存结果
    visual_prompt = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if save:
        cv2.imwrite(output_path, visual_prompt)
    return visual_prompt

    # 显示结果
    # cv2.imshow('Blurred Background', result_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    # 示例调用
    image_path = 'resources/input/image/test1.jpg'
    cam_path = 'resources/output/cam/phone.npy'
    threshold = 0.5  # 设置阈值
    output_path = 'resources/output/visual_prompt/phone.png'
    blur_background(image_path, cam_path, threshold, output_path)