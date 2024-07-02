import numpy as np
import cv2



def generate_heatmap(image_path, cam_dic, heatmap_save_path='', save=False):
    """
    用生成的CAM生成热图并保存

    :param image_path: 原始图像路径
    :param cam_dic: CAM字典
    :param heatmap_save_path: 保存热图的路径
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)  # 读取图像并转换为浮点型
    cam = cam_dic['attn_highres']

    # 去掉CAM的多余维度，使其维度变为（H，W）
    cam = cam.squeeze()

    # 归一化CAM
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    # 将CAM转换为热图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET).astype(np.float32)

    # 叠加热图和原始图片
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    heatmap = overlay.astype(np.uint8)
    # 保存热图
    if save:
        cv2.imwrite(heatmap_save_path, heatmap)
    return heatmap

    # # 显示热图
    # cv2.imshow('Heatmap', overlay.astype(np.uint8))
    # cv2.waitKey(0)  # 按任意键关闭窗口
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    generate_heatmap(image_path="resources/input/image/phone.jpg", cam_path="resources/output/cam/phone.npy", heatmap_save_path="resources/output/heatmap/phone.png")