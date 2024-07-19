import os

from generate_cam import generate_cam, zeroshot_classifier, reshape_transform, image_preprocess
from mask_proposal_generator import mask_proposal_generator
from mask_classifier import mask_classifier
from post_process import post_process
from clip_es.pytorch_grad_cam import GradCAM
import torch
from clip_es import clip
from clip_text import BACKGROUND_CATEGORY_COCO
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"  # 选择设备
model_path = 'resources/models/ViT-B-16.pt'
model, _ = clip.load(model_path, device=device)
# 获取背景文本特征
# bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY_COCO, ['a clean origami {}.'], model).to(device)
from clip_text import BACKGROUND_Terrestrial, BACKGROUND_Man_Made, BACKGROUND_Aquatic_Atmospheric
# background_query = BACKGROUND_Terrestrial + BACKGROUND_Man_Made + BACKGROUND_Aquatic_Atmospheric
background_query = BACKGROUND_CATEGORY_COCO
bg_text_features = zeroshot_classifier(background_query, ['a clean origami {}.'], model).to(device)

# 创建GradCAM对象
target_layers = [model.visual.transformer.resblocks[-1].ln_1]  # 设置目标层
cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)


# 将 NumPy 数组转换为 PIL 图像
def numpy2pil(image_list):
    new_image_list = []
    for image in image_list:
        new_image_list.append(Image.fromarray(image))
    return new_image_list


def showImage(final_heatmap, final_mask, image_path):
    # show image
    image = cv2.imread(image_path)
    # 创建一个 1x3 的子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    # 显示原图
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    # 显示最终热图
    ax2.imshow(cv2.cvtColor(final_heatmap, cv2.COLOR_BGR2RGB))
    ax2.set_title('Final Heatmap')
    ax2.axis('off')
    # 显示最终掩码
    ax3.imshow(final_mask, cmap='gray')
    ax3.set_title('Final Mask')
    ax3.axis('off')
    # 调整子图间距
    plt.subplots_adjust(wspace=0.5)
    # 显示图片
    plt.show()


def predict(image_path, output_path, text_query, background_query, eta, theta, save=False):
    label_list = text_query + background_query
    new_label_list = label_list
    iterator = 0  # 迭代次数
    cam_dic_list = []
    while len(new_label_list) > 0:
        iterator += 1
        print(f"第{iterator}次迭代：")
        print(new_label_list)
        # mask proposal
        cam_dic_list, visual_prompt_list = mask_proposal_generator(model=model, cam=cam,
                                                                   bg_text_features=bg_text_features,
                                                                   image_path=image_path, label_list=label_list,
                                                                   output_path=output_path, eta=eta, save=False)
        visual_prompt_list = numpy2pil(visual_prompt_list)
        # mask classifier
        new_label_list = mask_classifier(visual_prompt_list=visual_prompt_list, label_list=label_list, theta=theta)
        if len(label_list) == len(new_label_list):
            break
        label_list = new_label_list
        print()
    # post process
    final_heatmap, final_mask = post_process(image_path=image_path, cam_dics=cam_dic_list, save=False)
    if save:
        image_name = image_path.split("/")[-1].split(".")[0]
        cv2.imwrite(f"{output_path}/mask/{image_name}.png", final_mask)
    else:
        showImage(final_heatmap, final_mask, image_path)


def predict_coco14(image_path, output_path, text_query, background_query, eta, theta, save=False):
    label_list = text_query + background_query
    new_label_list = label_list
    iterator = 0  # 迭代次数
    cam_dic_list = []
    while len(new_label_list) > 0:
        iterator += 1
        print(f"第{iterator}次迭代：")
        print(new_label_list)
        # mask proposal
        cam_dic_list, visual_prompt_list = mask_proposal_generator(model=model, cam=cam,
                                                                   bg_text_features=bg_text_features,
                                                                   image_path=image_path, label_list=label_list,
                                                                   output_path=output_path, eta=eta, save=False)
        visual_prompt_list = numpy2pil(visual_prompt_list)
        # mask classifier
        new_label_list = mask_classifier(visual_prompt_list=visual_prompt_list, label_list=label_list, theta=theta)
        if len(label_list) == len(new_label_list):
            break
        label_list = new_label_list
        print()
    # post process
    # final_heatmap, final_mask = post_process(image_path=image_path, cam_dics=cam_dic_list, save=False)
    from post_process import post_process_coco14
    final_mask = post_process_coco14(image_path=image_path, cam_dics=cam_dic_list, save=False, label_list=text_query)
    if save:
        image_name = image_path.split("/")[-1].split(".")[0]

        img = Image.fromarray(final_mask.astype('uint8'))
        img.save(f"{output_path}/mask/{image_name}.png")
    else:
        # 创建自定义颜色映射
        cmap_colors = plt.cm.get_cmap('tab20b', 81).colors
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(cmap_colors)

        # 显示图像
        plt.figure(figsize=(8, 6))
        plt.imshow(final_mask, cmap=custom_cmap)
        plt.colorbar()
        plt.title('Customized Colormap')
        plt.show()


def run_coco14(dataset_dir, output_path, split_file, test_num=-1, start=0):
    read_num = 0
    with open(split_file, 'r') as f:
        lines = f.readlines()
    if test_num == -1:
        total_num =  len(lines)
    else:
        total_num = test_num
    from clip_text import new_class_names_coco
    for line in lines:
        # 判断是否超出要生成的图片张数
        read_num += 1
        if read_num < start:
            continue

        if test_num != -1:
            if read_num > test_num:
                break

        # process
        print(f"processing {read_num}/{total_num}.....................\n")
        items = line.strip().split()
        image_name = items[0]
        class_id_list = [int(x) for x in items[1:]]
        image_path = f'{dataset_dir}/JPEGImages/val2014/{image_name}.jpg'
        class_name_list = [new_class_names_coco[class_id] for class_id in class_id_list]
        background_query = []
        predict_coco14(image_path=image_path, output_path=output_path, text_query=class_name_list,
                background_query=background_query, eta=0.5, theta=0.3, save=True)


def main_test():
    # 参数设置
    eta = 0.5
    theta = 0.3
    # 输入输出设置
    # 输出路径
    output_path = "resources/output"
    # 图片路径
    image_path = 'resources/input/image/COCO_val2014_000000000139.jpg'
    # 定义标签
    from clip_text import new_class_names_coco

    text_query = [new_class_names_coco[i] for i in [0, 56, 58, 60, 62, 68, 72, 73, 74, 75]]
    background_query = []
    predict_coco14(image_path=image_path, output_path=output_path, text_query=text_query, background_query=background_query, eta=eta, theta=theta, save=False)

def main_coco14():
    dataset_dir = "D:\et\program\code\python\zju\dataset\coco2014"
    output_path = "D:\et\program\code\python\zju\dataset\coco2014\MyResult1"
    split_file = "eval_resources/coco14/val.txt"
    run_coco14(dataset_dir, output_path, split_file, test_num=-1, start=2432)

if __name__ == '__main__':
    main_coco14()