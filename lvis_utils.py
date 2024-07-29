import json

from lvis import LVIS
import os
import shutil

# 加载LVIS注释文件
lvis = LVIS('D:\et\program\code\python\zju\dataset\lvis\lvis_v1_val.json')

# # 获取所有类别ID
# category_ids = lvis.get_cat_ids()
#
# # 加载所有类别信息
# categories = lvis.load_cats(category_ids)
# print("Categories:", categories)
# 提取类别名称
# category_names = [category['name'] for category in categories]
#
# print(category_names)

# 获取所有图片ID{'date_captured': '2013-11-19 21:50:58', 'neg_category_ids': [562, 350, 740, 615, 641, 600], 'id': 159495, 'license': 1, 'height': 500, 'width': 375, 'flickr_url': 'http://farm4.staticflickr.com/3589/3498178415_a5c5b41636_z.jpg', 'coco_url': 'http://images.cocodataset.org/train2017/000000159495.jpg', 'not_exhaustive_category_ids': [45]}
# image_ids = lvis.get_img_ids()
# dir_path = 'D:\et\program\code\python\zju\dataset\lvis\\train2017\\'
# file_paths = [dir_path + str(image_id).zfill(12) + '.jpg' for image_id in image_ids]
# print("Image IDs:", file_paths)
# dst_dir = 'D:\et\program\code\python\zju\dataset\lvis\lvis_val'
# for file_path in file_paths:
#     file_name = os.path.basename(file_path)
#     dst_path = os.path.join(dst_dir, file_name)
#     shutil.copy2(file_path, dst_path)
#     print(f"Copied '{file_path}' to '{dst_path}'")
# 加载所有图片的信息
# images = lvis.load_imgs(image_ids)
# print("Images:", images)
#
# 获取所有实例分割注释
# {'area': 79.39, 'id': 5650, 'segmentation': [[153.61, 11.28, 150.33, 8.61, 148.9, 4.71, 149.1, 1.64, 149.51, 0.4, 153.2, 0.81, 152.59, 3.07, 152.79, 5.54, 153.61, 7.59, 153.61, 11.28], [160.79, 11.28, 162.85, 9.44, 164.08, 4.92, 164.69, 0.61, 161.0, 0.61, 161.2, 4.3, 160.18, 6.15, 157.72, 7.59, 156.9, 9.44, 157.31, 11.9, 160.79, 11.28]], 'image_id': 397133, 'bbox': [148.9, 0.4, 15.79, 11.5], 'category_id': 566}
# ann_ids = lvis.get_ann_ids(img_ids=[397133])
# anns = lvis.load_anns(ann_ids)
# print("Annotations:", anns)

# 读取 JSON 文件
json_dir = 'resources/output/lvis/'
file_list = os.listdir(json_dir)
results = []
for file in file_list:
    file_path = os.path.join(json_dir, file)
    with open(file_path, 'r') as f:
        results = results + json.load(f)

# with open('resources/output/lvis/results.json', 'r') as f:
#     results = json.load(f)
#
# for result in results:
#     result['image_id'] = int(result['image_id'])
with open('resources/output/lvis/results.json', 'w') as f:
    json.dump(results, f)