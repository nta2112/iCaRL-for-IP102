import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from config import find_image_dir


def load_coco_split(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    file_name = {im['id']: im['file_name'] for im in d['images']}
    return file_name, d['annotations']


def build_image_label(anns):
    image_id_to_cat = {}
    for a in anns:
        img = a['image_id']
        cat = a['category_id']
        area = a.get('area', 0)
        if img not in image_id_to_cat or area > image_id_to_cat[img][1]:
            image_id_to_cat[img] = (cat, area)
    return {k: v[0] for k, v in image_id_to_cat.items()}


class iIP102(Dataset):

    def __init__(self, data_root, transform=None, test_transform=None,
                 target_transform=None, class_ids=None, cache_images=True,
                 cache_size=256):
        super(iIP102, self).__init__()
        self.data_root = data_root
        self.image_dir = find_image_dir(data_root)
        self.transform = transform
        self.test_transform = test_transform
        self.target_transform = target_transform
        self.cache_images = cache_images
        self.cache_size = cache_size

        self.meta = {}
        for s in ('train', 'test', 'val'):
            jp = os.path.join(data_root, s + '.json')
            self.meta[s] = load_coco_split(jp) if os.path.exists(jp) else (None, None)

        if class_ids is None:
            class_ids = sorted(set(a['category_id'] for a in self.meta['train'][1]))
        self.class_ids = list(class_ids)
        self.class_id_to_idx = {cid: i for i, cid in enumerate(self.class_ids)}
        self.num_classes = len(self.class_ids)

        for s in ('train', 'test', 'val'):
            file_name, anns = self.meta[s]
            if file_name is None:
                continue
            img_label = build_image_label(anns)
            by_cls = {i: [] for i in range(self.num_classes)}
            for img_id, cat in img_label.items():
                if cat in self.class_id_to_idx:
                    by_cls[self.class_id_to_idx[cat]].append(file_name[img_id])
            setattr(self, '_%s_by_cls' % s, by_cls)

        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

        self._cache = {}

    def _read_image(self, file_name, target_size=None):
        path = os.path.join(self.image_dir, file_name)
        img = Image.open(path).convert('RGB')
        if target_size is not None:
            img = img.resize((target_size, target_size), Image.BILINEAR)
        return np.array(img)

    def get_image_class(self, label):
        if label in self._cache:
            return self._cache[label]
        files = self._train_by_cls[label]
        arrays = np.stack([self._read_image(f, self.cache_size) for f in files])
        if self.cache_images:
            self._cache[label] = arrays
        return arrays

    def getTrainData(self, classes, exemplar_set):
        datas, labels = [], []
        if len(exemplar_set) != 0:
            for j, ex in enumerate(exemplar_set):
                datas.append(np.asarray(ex))
                labels.append(np.full(len(ex), j))
        for label in range(classes[0], classes[1]):
            arr = self.get_image_class(label)
            datas.append(arr)
            labels.append(np.full(arr.shape[0], label))
        self.TrainData = np.concatenate(datas, axis=0)
        self.TrainLabels = np.concatenate(labels, axis=0).astype(np.int64)
        print('the size of train set is %s' % (str(self.TrainData.shape)))

    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            files = self._test_by_cls[label]
            arr = np.stack([self._read_image(f, self.cache_size) for f in files])
            datas.append(arr)
            labels.append(np.full(arr.shape[0], label))
        new_data = np.concatenate(datas, axis=0)
        new_labels = np.concatenate(labels, axis=0).astype(np.int64)
        if len(self.TestData) == 0:
            self.TestData = new_data
            self.TestLabels = new_labels
        else:
            self.TestData = np.concatenate([self.TestData, new_data], axis=0)
            self.TestLabels = np.concatenate([self.TestLabels, new_labels], axis=0)
        print('the size of test set is %s' % (str(self.TestData.shape)))

    def _get_item(self, data, labels, index, trf):
        img = Image.fromarray(data[index])
        target = labels[index]
        if trf is not None:
            img = trf(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return index, img, target

    def __getitem__(self, index):
        if len(self.TrainData) != 0:
            return self._get_item(self.TrainData, self.TrainLabels, index, self.transform)
        elif len(self.TestData) != 0:
            return self._get_item(self.TestData, self.TestLabels, index, self.test_transform)

    def __len__(self):
        if len(self.TrainData) != 0:
            return len(self.TrainData)
        elif len(self.TestData) != 0:
            return len(self.TestData)
        return 0
