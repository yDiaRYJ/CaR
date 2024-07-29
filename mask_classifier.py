import torch
import clip
from PIL import Image
import numpy as np
from generate_cam import zeroshot_classifier

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device, download_root="resources/models/")


def get_similarity_matrix(images, texts):
    # 预处理图像和文本
    image_inputs = torch.stack([preprocess(img).to(device) for img in images])

    # 计算图像
    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # 计算文本特征
    text_features = zeroshot_classifier(texts, ['a photo of {}.'], model)

    # similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    similarity_matrix = (100.0 * image_features @ text_features.T)
    similarity_matrix = similarity_matrix.softmax(dim = 1)
    return similarity_matrix


def mask_classifier(visual_prompt_list, label_list, theta=0.3):
    # tokenizer = clip.tokenize
    # logits = torch.empty(0, 1).to(device)
    # for process_image, text in zip(visual_prompt_list, label_list):
    #     process_image = preprocess(process_image).unsqueeze(0).to(device)
    #     text = tokenizer(text).to(device)
    #
    #     with torch.no_grad():
    #         logits_per_image, logits_per_text = model(process_image, text)
    #         logits = torch.cat([logits, logits_per_image], dim=0)
    #
    #
    # probs = logits.softmax(dim=0).cpu().numpy()
    # print("probs:")
    # print(probs)
    # new_label_list = []
    # for i, score in enumerate(probs):
    #     if score >= theta:
    #         new_label_list.append(label_list[i])
    # print(f"迭代后：{new_label_list}")
    # return new_label_list
    threshold = 1 / len(label_list)
    if threshold > theta:
        threshold = theta
    similarity_matrix = get_similarity_matrix(visual_prompt_list, label_list)
    # 提取对角线元素作为匹配分数
    matching_scores = torch.diag(similarity_matrix)
    print("matching_scores:")
    print(matching_scores)
    # 根据阈值进行文本筛选
    new_label_list = []
    for i, score in enumerate(matching_scores):
        if score >= threshold:
            new_label_list.append(label_list[i])
    print(f"迭代后：{new_label_list}")
    return new_label_list


if __name__ == '__main__':
    images = []
    image_name = "test1"
    label_list = ['person with clothes,people,human', 'cell phone', 'face', 'building', 'dog']
    for label in label_list:
        image = f"resources/output/visual_prompt/{image_name}/{label}.png"
        images.append(image)

    similarity_matrix = get_similarity_matrix(images, label_list)
