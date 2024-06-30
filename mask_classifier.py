import torch
import clip
from PIL import Image

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device, download_root="resources/models/")


def get_similarity_matrix(images, texts):
    # 预处理图像和文本
    image_inputs = torch.stack([preprocess(Image.open(img)).to(device) for img in images])
    text_inputs = clip.tokenize([txt for txt in texts]).to(device)

    # 计算图像和文本特征
    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)

    # 计算相似度矩阵
    similarity_matrix = image_features @ text_features.T
    return similarity_matrix


def filter_queries(similarity_matrix, threshold=0.3):
    # 沿文本维度应用 softmax（按列进行softmax）
    softmax_sim_matrix = similarity_matrix.softmax(dim=0)

    # 提取对角线元素作为匹配分数
    matching_scores = torch.diag(softmax_sim_matrix)

    # 应用阈值进行过滤
    filtered_queries = []
    for i, score in enumerate(matching_scores):
        if score >= threshold:
            print(f"查询 {label_list[i]} 得分：{score}，保留")
            filtered_queries.append((i, score.item()))  # (索引, 分数)
        else:
            print(f"查询 {label_list[i]} 得分：{score}，过滤")
            filtered_queries.append((i, None))  # 标记为 NULL
    return filtered_queries

if __name__ == '__main__':
    images = []
    image_name = "test1"
    label_list = ['face', 'human', 'people', 'person with clothes', 'phone']
    for label in label_list:
        image = f"resources/output/visual_prompt/{image_name}/{label}.png"
        images.append(image)

    similarity_matrix = get_similarity_matrix(images, label_list)
    filtered_queries = filter_queries(similarity_matrix, threshold=0.3)