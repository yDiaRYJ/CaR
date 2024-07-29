import torch
import clip
from PIL import Image

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device, download_root="resources/models/")

def pre_process_text(image_path, texts, topk):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    # 将文本编码为向量
    text_inputs = clip.tokenize(texts).to(device)
    # 计算图像和文本的特征向量
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    # 计算相似度
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # 获取最相似的100个文本及其概率
    top_probs, top_indices = similarity[0].topk(topk)
    # 获取最相似的文本列表
    top_texts = [texts[idx] for idx in top_indices]
    return top_texts


if __name__ == '__main__':
    # 预处理图像
    image_path = "resources/input/image/000000000139.jpg"  # 替换为你的图像路径
    # 文本列表
    from lvis_text import class_names_lvis
    texts = class_names_lvis
    top_texts = pre_process_text(image_path, texts, topk=100)
    # 打印最相似的文本及其概率
    print(top_texts)