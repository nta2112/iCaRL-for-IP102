# =============================================================================
# iCaRL for IP102 (25 classes, >200 images/class) - SINGLE FILE KAGGLE SCRIPT
#
# Cach dung:
#   - Kaggle Notebook: paste toan bo file nay vao mot cell va chay.
#     Dataset IP102 (chua train.json / test.json / val.json va thu muc
#     VOC2007/VOC2007/JPEGImages) duoc add vao Input voi ten bat ky.
#   - Hoac chay truc tiep:  python kaggle_ip102_icarl.py
#
# Mo hinh: ResNet18 (ImageNet pretrained) + CBAM
# Chia task: 7 lop (task 0) + 6 lop (task 1) + 6 lop (task 2) + 6 lop (task 3)
# Anh dau vao: 224x224
# =============================================================================

import copy
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ============================ CONFIG =========================================
def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


IMG_SIZE = _env_int('IP102_IMG_SIZE', 224)      # kich thuoc anh dau vao
CACHE_SIZE = _env_int('IP102_CACHE_SIZE', 256)  # kich thuoc cache anh (truoc khi crop)
BATCH_SIZE = _env_int('IP102_BATCH_SIZE', 32)
MEMORY_SIZE = _env_int('IP102_MEMORY_SIZE', 2000)   # tong so exemplar
EPOCHS = _env_int('IP102_EPOCHS', 1)
LEARNING_RATE = _env_float('IP102_LR', 2.0)
LR_FRACTIONS = [0.48, 0.62, 0.80]
LR_MULTIPLIERS = [0.2, 0.04, 0.008]
TASK_SIZES = [7, 6, 6, 6]
MAX_TASKS = _env_int('IP102_MAX_TASKS', len(TASK_SIZES))  # chi chay N task dau (test nhanh)
PRETRAINED = True
SEED = _env_int('IP102_SEED', 0)


def find_data_root():
    if os.environ.get('IP102_DATA_ROOT'):
        root = os.environ['IP102_DATA_ROOT']
        if os.path.exists(root):
            return root
    if os.path.isdir('/kaggle/input'):
        found = _find_dir_with_file('/kaggle/input', 'train.json', maxdepth=6)
        if found:
            return found
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in [os.path.join(here, 'IP102 dataset'), here]:
        if os.path.exists(os.path.join(cand, 'train.json')):
            return cand
    raise FileNotFoundError('Khong tim thay thu muc dataset (train.json)')


def _find_dir_with_file(base, filename, maxdepth=6):
    base = os.path.abspath(base)
    for dirpath, dirnames, filenames in os.walk(base):
        depth = dirpath[len(base):].count(os.sep)
        if depth > maxdepth:
            dirnames[:] = []
            continue
        if filename in filenames:
            return dirpath
    return None


def find_image_dir(data_root):
    for rel in ['VOC2007/VOC2007/JPEGImages', 'VOC2007/JPEGImages',
                'JPEGImages', 'images', 'Images']:
        p = os.path.join(data_root, rel)
        if os.path.isdir(p):
            return p
    for dirpath, dirnames, filenames in os.walk(data_root):
        if os.path.basename(dirpath).lower() in ('jpegimages', 'images'):
            return dirpath
    raise FileNotFoundError('Khong tim thay thu muc JPEGImages trong ' + data_root)


def get_class_ids(data_root):
    with open(os.path.join(data_root, 'train.json'), 'r', encoding='utf-8') as f:
        d = json.load(f)
    return sorted(set(a['category_id'] for a in d['annotations']))


DATA_ROOT = find_data_root()
IMAGE_DIR = find_image_dir(DATA_ROOT)
CLASS_IDS = get_class_ids(DATA_ROOT)
SAVE_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'output')


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

    def get_class_images(self, split, label):
        key = (split, label)
        if key in self._cache:
            return self._cache[key]
        by_cls = getattr(self, '_%s_by_cls' % split)
        files = by_cls[label]
        arrays = np.stack([self._read_image(f, self.cache_size) for f in files])
        if self.cache_images:
            self._cache[key] = arrays
        return arrays

    def get_image_class(self, label):
        return self.get_class_images('train', label)

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


# ============================ NETWORK ========================================
class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)

    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self, inputs):
        return self.feature(inputs)


# ============================ RESNET18 + CBAM ================================
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 1. / math.sqrt(n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x


def _adapt_pretrained_state_dict(model, sd):
    model_sd = model.state_dict()
    filtered = {}
    for k, v in sd.items():
        if k not in model_sd:
            continue
        if tuple(v.shape) == tuple(model_sd[k].shape):
            filtered[k] = v
        elif k == 'conv1.weight' and v.ndim == 4 \
                and tuple(v.shape[1:]) == (3, 7, 7) \
                and tuple(model_sd[k].shape[1:]) == (3, 3, 3):
            print('adapting conv1.weight 7x7 -> 3x3 (center crop)')
            filtered[k] = v[:, :, 2:5, 2:5].contiguous()
        else:
            print('skip pretrained key %s (shape %s != model %s)'
                  % (k, tuple(v.shape), tuple(model_sd[k].shape)))
    return filtered


def _load_pretrained(model):
    try:
        import torchvision.models as tvm
        try:
            sd = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1).state_dict()
        except Exception:
            sd = tvm.resnet18(pretrained=True).state_dict()
    except Exception:
        try:
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
        except Exception as e:
            raise RuntimeError('Could not load ImageNet pretrained weights: %s' % e)
    filtered = _adapt_pretrained_state_dict(model, sd)
    model.load_state_dict(filtered, strict=False)
    return model


def resnet18_cbam(pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model = _load_pretrained(model)
    return model


# ============================ EVALUATION METRICS =============================
def calculate_auroc_numpy(y_true, y_scores):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    desc = np.argsort(y_scores)[::-1]
    y_true = y_true[desc]
    y_scores = y_scores[desc]
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    tpr = tp / n_pos
    fpr = fp / n_neg
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    area = 0.0
    for i in range(len(fpr) - 1):
        area += 0.5 * (tpr[i] + tpr[i + 1]) * (fpr[i + 1] - fpr[i])
    return area, fpr, tpr


def calculate_fpr_at_tpr95(fpr, tpr):
    idx = np.where(tpr >= 0.95)[0]
    return fpr[idx[0]] if len(idx) > 0 else 1.0


def compute_ap(q_class, ranked_classes):
    ap = 0.0
    hits = 0
    total_pos = sum(1 for g_class in ranked_classes if g_class == q_class)
    if total_pos == 0:
        return 0.0
    for rank, g_class in enumerate(ranked_classes):
        if g_class == q_class:
            hits += 1
            ap += hits / (rank + 1)
            if hits == total_pos:
                break
    return ap / total_pos


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def evaluate_retrieval(q_embeds, q_classes, g_embeds, g_classes):
    """Returns (class_metrics, macro_r1, macro_r5, macro_r10, macro_ap, q_aps, q_r1s)."""
    q_embeds = l2_normalize(q_embeds)
    g_embeds = l2_normalize(g_embeds)
    q_classes = np.asarray(q_classes)
    g_classes = np.asarray(g_classes)
    sims = q_embeds @ g_embeds.T
    n = len(q_embeds)
    r1s = np.zeros(n)
    r5s = np.zeros(n)
    r10s = np.zeros(n)
    aps = np.zeros(n)
    for i in range(n):
        ranked = g_classes[np.argsort(sims[i])[::-1]]
        qc = q_classes[i]
        r1s[i] = 1.0 if qc in ranked[:1] else 0.0
        r5s[i] = 1.0 if qc in ranked[:5] else 0.0
        r10s[i] = 1.0 if qc in ranked[:10] else 0.0
        aps[i] = compute_ap(qc, ranked)
    class_metrics = {}
    for c in np.unique(q_classes):
        mask = q_classes == c
        class_metrics[int(c)] = {
            'count': int(mask.sum()),
            'R@1': float(np.mean(r1s[mask])),
            'R@5': float(np.mean(r5s[mask])),
            'R@10': float(np.mean(r10s[mask])),
            'AP': float(np.mean(aps[mask])),
        }
    if not class_metrics:
        return {}, 0.0, 0.0, 0.0, 0.0, aps, r1s
    macro_r1 = np.mean([class_metrics[c]['R@1'] for c in class_metrics])
    macro_r5 = np.mean([class_metrics[c]['R@5'] for c in class_metrics])
    macro_r10 = np.mean([class_metrics[c]['R@10'] for c in class_metrics])
    macro_ap = np.mean([class_metrics[c]['AP'] for c in class_metrics])
    return (class_metrics, float(macro_r1), float(macro_r5), float(macro_r10),
            float(macro_ap), aps, r1s)


def evaluate_ood(q_embeds, q_classes, seen_class_means, seen_classes):
    """Unseen classes are 'unknown'; unknown score = -max cos sim to seen class means."""
    q_embeds = l2_normalize(q_embeds)
    seen_class_means = l2_normalize(np.asarray(seen_class_means))
    sims = q_embeds @ seen_class_means.T
    unknown_score = -sims.max(axis=1)
    seen_set = set(int(c) for c in seen_classes)
    y_true = np.array([0 if int(c) in seen_set else 1 for c in q_classes])
    num_pos = int(y_true.sum())
    num_neg = len(y_true) - num_pos
    if num_pos > 0 and num_neg > 0:
        auroc, fpr, tpr = calculate_auroc_numpy(y_true, unknown_score)
        return float(auroc), float(calculate_fpr_at_tpr95(fpr, tpr))
    return None, None


# ============================ iCaRL MODEL ====================================
def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(target.device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


class iCaRLmodel:

    def __init__(self, numclass, feature_extractor, batch_size, task_size,
                 memory_size, epochs, learning_rate, device):
        super(iCaRLmodel, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        base_model = network(numclass, feature_extractor)
        self.use_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
        self.model = nn.DataParallel(base_model) if self.use_multi_gpu else base_model
        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = numclass
        self.task_size = task_size
        self.batchsize = batch_size
        self.memory_size = memory_size

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            self.norm])

        self.train_transform = transforms.Compose([
            transforms.Resize(CACHE_SIZE),
            transforms.RandomCrop((IMG_SIZE, IMG_SIZE), padding=8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25),
            transforms.ToTensor(),
            self.norm])

        self.test_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            self.norm])

        self.classify_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=1.),
            transforms.ToTensor(),
            self.norm])

        self.train_dataset = iIP102(DATA_ROOT, transform=self.train_transform,
                                    class_ids=CLASS_IDS, cache_size=CACHE_SIZE)
        self.test_dataset = iIP102(DATA_ROOT, test_transform=self.test_transform,
                                   class_ids=CLASS_IDS, cache_size=CACHE_SIZE)

        self.train_loader = None
        self.test_loader = None
        self.old_model = None
        self.save_dir = SAVE_DIR
        self.history = {}
        self.history_path = os.path.join(self.save_dir, 'history.json')
        self._load_history()
        self.task_of_class = {}
        offset = 0
        for tid, sz in enumerate(TASK_SIZES):
            for c in range(offset, offset + sz):
                self.task_of_class[c] = tid
            offset += sz

    def _load_history(self):
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
        except Exception as e:
            print('could not load history: %s' % e)

    def beforeTrain(self):
        self.model.eval()
        classes = [self.numclass - self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if self.numclass > self.task_size:
            self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(self.device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        n_devices = torch.cuda.device_count() if self.use_multi_gpu else 1
        bs = self.batchsize
        while bs > n_devices and (len(self.test_dataset) % bs) in range(1, n_devices):
            bs -= 1
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize,
                                  drop_last=self.use_multi_gpu)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=False,
                                 batch_size=bs)
        return train_loader, test_loader

    def train(self):
        accuracy = 0
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate,
                        momentum=0.9, nesterov=True, weight_decay=0.00001)
        milestones = [e for e in (int(self.epochs * f) for f in LR_FRACTIONS) if e > 0]
        for epoch in range(self.epochs):
            if epoch in milestones:
                idx = milestones.index(epoch)
                new_lr = self.learning_rate * LR_MULTIPLIERS[idx]
                for p in opt.param_groups:
                    p['lr'] = new_lr
                print('change learning rate to:%.3f' % new_lr)
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)
                loss_value = self._compute_loss(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()
            accuracy = self._test(self.test_loader, 1)
            print('epoch:%d, accuracy:%.3f' % (epoch, accuracy))
        return accuracy

    def _test(self, testloader, mode):
        self.model.eval()
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy

    def _compute_loss(self, indexs, imgs, target):
        output = self.model(imgs)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(self.device), target.to(self.device)
        if self.old_model is None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            old_target = torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)

    def afterTrain(self, accuracy, task_id):
        self.model.eval()
        m = int(self.memory_size / self.numclass)
        self._reduce_exemplar_sets(m)
        for i in range(self.numclass - self.task_size, self.numclass):
            print('construct class %s examplar:' % (i), end='')
            images = self.train_dataset.get_image_class(i)
            self._construct_exemplar_set(images, m)
        self.compute_exemplar_class_mean()
        self.model.train()
        KNN_accuracy = self._test(self.test_loader, 0)
        print('NMS accuracy:' + str(KNN_accuracy.item()))

        base = self.model.module if self.use_multi_gpu else self.model
        self.old_model = copy.deepcopy(base).to(self.device)
        if self.use_multi_gpu:
            self.old_model = nn.DataParallel(self.old_model)
        self.old_model.eval()

        os.makedirs(self.save_dir, exist_ok=True)
        filename = os.path.join(
            self.save_dir,
            'icarl_ip102_task%d_acc%.3f_knn%.3f.pkl' % (task_id, accuracy, KNN_accuracy))
        sd = self.model.module.state_dict() if self.use_multi_gpu else self.model.state_dict()
        torch.save({'state_dict': sd,
                    'numclass': self.numclass,
                    'class_ids': CLASS_IDS}, filename)
        print('model saved to %s' % filename)
        self.evaluate_all(task_id, accuracy, KNN_accuracy)

    def _embed(self, image_arrays):
        chunk = 64
        if self.use_multi_gpu:
            n_devices = torch.cuda.device_count()
            while chunk > n_devices and (len(image_arrays) % chunk) in range(1, n_devices):
                chunk -= 1
        outs = []
        for i in range(0, len(image_arrays), chunk):
            batch = self.Image_transform(image_arrays[i:i + chunk], self.test_transform).to(self.device)
            with torch.no_grad():
                feats = F.normalize(self.model.feature_extractor(batch).detach())
            outs.append(feats.cpu().numpy())
        if not outs:
            return np.zeros((0, 512), dtype=np.float32)
        return np.concatenate(outs, axis=0)

    def _split_images(self, split, classes):
        arrays, labels = [], []
        for c in classes:
            arr = self.train_dataset.get_class_images(split, c)
            arrays.append(arr)
            labels.append(np.full(len(arr), c))
        return np.concatenate(arrays, axis=0), np.concatenate(labels).astype(int)

    def evaluate_all(self, task_id, accuracy, knn):
        seen = list(range(self.numclass))
        total_classes = list(range(len(CLASS_IDS)))

        val_all, val_labels = self._split_images('val', total_classes)
        val_emb = self._embed(val_all)

        test_all, test_labels = self._split_images('test', seen)
        test_emb = self._embed(test_all)

        seen_mask = np.array([c in seen for c in val_labels])
        q_emb, q_cls = val_emb[seen_mask], val_labels[seen_mask]
        _, macro_r1, macro_r5, macro_r10, macro_ap, q_aps, q_r1s = \
            evaluate_retrieval(q_emb, q_cls, test_emb, test_labels)

        task_ids = sorted(set(self.task_of_class[c] for c in seen))
        group_mAP = {}
        for t in task_ids:
            m = np.array([self.task_of_class[c] == t for c in q_cls])
            if m.sum() > 0:
                group_mAP[t] = float(np.mean(q_aps[m]))

        means = np.array(self.class_mean_set)
        auroc, fpr95 = evaluate_ood(val_emb, val_labels, means, seen)

        self.history[str(task_id)] = {str(t): round(group_mAP[t], 6) for t in group_mAP}
        self._save_history()

        plasticity = group_mAP.get(task_id, 0.0)
        forgets = []
        for t in range(task_id):
            if str(t) in self.history and str(task_id) in self.history:
                peak = self.history[str(t)].get(str(t), 0.0)
                curr = self.history[str(task_id)].get(str(t), 0.0)
                forgets.append(max(0.0, peak - curr))
        forgetting = float(np.mean(forgets)) if forgets else 0.0
        overall = plasticity - forgetting

        print('=== Task %d evaluation ===' % task_id)
        print('Retrieval (seen classes): R@1 %.3f | R@5 %.3f | R@10 %.3f | mAP %.3f'
              % (macro_r1, macro_r5, macro_r10, macro_ap))
        print('OOD AUROC %.3f | FPR@TPR95 %.3f' % ((auroc or 0.0), (fpr95 or 0.0)))
        print('Plasticity %.3f | Forgetting %.3f | Overall %.3f'
              % (plasticity, forgetting, overall))

        self._log_result(task_id, accuracy, knn, macro_r1, macro_r5, macro_r10,
                         macro_ap, auroc, fpr95, plasticity, forgetting, overall)

    def _save_history(self):
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print('could not save history: %s' % e)

    def _log_result(self, task_id, accuracy, knn, r1=0.0, r5=0.0, r10=0.0,
                    mAP=0.0, auroc=None, fpr95=None, plasticity=0.0,
                    forgetting=0.0, overall=0.0):
        path = os.path.join(self.save_dir, 'results.csv')
        header = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if header:
                w.writerow(['task', 'numclass', 'softmax_acc', 'knn_acc',
                            'R@1', 'R@5', 'R@10', 'mAP',
                            'AUROC', 'FPR95', 'Plasticity', 'Forgetting', 'Overall'])
            w.writerow([task_id, self.numclass,
                        round(float(accuracy), 3), round(float(knn), 3),
                        round(float(r1), 3), round(float(r5), 3), round(float(r10), 3),
                        round(float(mAP), 3),
                        '-' if auroc is None else round(float(auroc), 3),
                        '-' if fpr95 is None else round(float(fpr95), 3),
                        round(float(plasticity), 3), round(float(forgetting), 3),
                        round(float(overall), 3)])

    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, feature_extractor_output.shape[1]))
        for i in range(m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])
        print('the size of exemplar :%s' % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))

    def Image_transform(self, images, transform):
        tensors = [transform(Image.fromarray(img)) for img in images]
        return torch.stack(tensors)

    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).to(self.device)
        feature_extractor_output = F.normalize(
            self.model.feature_extractor(x).detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            print('compute the class mean of %s' % (str(index)))
            exemplar = self.exemplar_set[index]
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_, _ = self.compute_class_mean(exemplar, self.classify_transform)
            class_mean = (class_mean / np.linalg.norm(class_mean) +
                          class_mean_ / np.linalg.norm(class_mean_)) / 2
            self.class_mean_set.append(class_mean)

    def classify(self, test):
        result = []
        test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)


# ============================ MAIN ===========================================
def main():
    torch.manual_seed(SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('DATA_ROOT:', DATA_ROOT)
    print('IMAGE_DIR:', IMAGE_DIR)
    print('num_classes:', len(CLASS_IDS), CLASS_IDS)
    print('task_sizes:', TASK_SIZES, '=> tasks:', len(TASK_SIZES),
          '(chay %d task dau)' % min(MAX_TASKS, len(TASK_SIZES)))
    print('device:', device, '| gpus:', torch.cuda.device_count() if torch.cuda.is_available() else 0)
    print('batch_size=%d epochs=%d lr=%.2f memory=%d img=%d' %
          (BATCH_SIZE, EPOCHS, LEARNING_RATE, MEMORY_SIZE, IMG_SIZE))

    feature_extractor = resnet18_cbam(pretrained=PRETRAINED)
    model = iCaRLmodel(numclass=TASK_SIZES[0],
                       feature_extractor=feature_extractor,
                       batch_size=BATCH_SIZE,
                       task_size=TASK_SIZES[0],
                       memory_size=MEMORY_SIZE,
                       epochs=EPOCHS,
                       learning_rate=LEARNING_RATE,
                       device=device)

    num_tasks = min(MAX_TASKS, len(TASK_SIZES))
    wall0 = time.time()
    for t in range(num_tasks):
        tsize = TASK_SIZES[t]
        model.numclass = sum(TASK_SIZES[:t + 1])
        model.task_size = tsize
        t0 = time.time()
        print('==== start task %d (classes %d..%d) ====' %
              (t, model.numclass - tsize, model.numclass))
        model.beforeTrain()
        accuracy = model.train()
        model.afterTrain(accuracy, t)
        print('==== task %d done, softmax acc %.3f, time %.1f s ====' %
              (t, accuracy, time.time() - t0))
    print('==== done: %d/%d tasks, total time %.1f s ====' %
          (num_tasks, len(TASK_SIZES), time.time() - wall0))


if __name__ == '__main__':
    main()
