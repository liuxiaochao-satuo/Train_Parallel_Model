import os
import json
import random

root = '/home/satuo/mmpose/data'
ann_dir = os.path.join(root, 'coco_annotations')

all_images = []
all_annotations = []
categories = None

img_id = 0
ann_id = 0

files = sorted(os.listdir(ann_dir))
for f in files:
    if not f.endswith('.json'):
        continue
    path = os.path.join(ann_dir, f)
    with open(path, 'r') as fp:
        data = json.load(fp)

    if categories is None:
        categories = data['categories']  # 只用第一份

    # 每个小 json 只有一张图，这里取 images[0]
    img = data['images'][0]
    img['id'] = img_id
    all_images.append(img)

    for ann in data['annotations']:
        ann['image_id'] = img_id
        ann['id'] = ann_id
        ann_id += 1
        all_annotations.append(ann)

    img_id += 1

# 打乱，划分 train / val
indices = list(range(len(all_images)))
random.shuffle(indices)

ratio = 0.8  # 80% 训练，20% 验证
train_num = int(len(indices) * ratio)
train_idx = set(indices[:train_num])

train_images, val_images = [], []
train_ann, val_ann = [], []

imgid_map_split = {}

for idx, img in enumerate(all_images):
    if idx in train_idx:
        train_images.append(img)
        imgid_map_split[img['id']] = 'train'
    else:
        val_images.append(img)
        imgid_map_split[img['id']] = 'val'

for ann in all_annotations:
    if imgid_map_split[ann['image_id']] == 'train':
        train_ann.append(ann)
    else:
        val_ann.append(ann)

out_dir = os.path.join(root, 'coco_parallel', 'annotations')
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, 'person_keypoints_train_parallel.json'), 'w') as f:
    json.dump({'categories': categories, 'images': train_images, 'annotations': train_ann}, f)

with open(os.path.join(out_dir, 'person_keypoints_val_parallel.json'), 'w') as f:
    json.dump({'categories': categories, 'images': val_images, 'annotations': val_ann}, f)