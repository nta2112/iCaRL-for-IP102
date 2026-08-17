import os
import json


def find_data_root():
    if os.environ.get('IP102_DATA_ROOT'):
        return os.environ['IP102_DATA_ROOT']
    if os.path.isdir('/kaggle/input'):
        for name in sorted(os.listdir('/kaggle/input')):
            base = os.path.join('/kaggle/input', name)
            if not os.path.isdir(base):
                continue
            if os.path.exists(os.path.join(base, 'train.json')):
                return base
            for sub in os.listdir(base):
                cand = os.path.join(base, sub)
                if os.path.isdir(cand) and os.path.exists(os.path.join(cand, 'train.json')):
                    return cand
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in [os.path.join(here, 'IP102 dataset'), here]:
        if os.path.exists(os.path.join(cand, 'train.json')):
            return cand
    raise FileNotFoundError('Khong tim thay thu muc dataset (train.json)')


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
    jp = os.path.join(data_root, 'train.json')
    with open(jp, 'r', encoding='utf-8') as f:
        d = json.load(f)
    ids = sorted(set(a['category_id'] for a in d['annotations']))
    return ids


class Config:
    data_root = find_data_root()
    image_dir = find_image_dir(data_root)
    class_ids = get_class_ids(data_root)

    is_kaggle = os.path.isdir('/kaggle/working')
    output_dir = '/kaggle/working' if is_kaggle else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output')

    img_size = 224
    cache_size = 256

    batch_size = 32
    memory_size = 2000
    epochs = 1
    learning_rate = 2.0

    lr_fractions = [0.48, 0.62, 0.80]
    lr_multipliers = [0.2, 0.04, 0.008]

    task_sizes = [7, 6, 6, 6]

    pretrained = True
    device = 'cuda:0'
    seed = 0

    def __str__(self):
        return ('Config(data_root=%s\n  class_ids=%s\n  num_classes=%d\n'
                '  task_sizes=%s\n  img_size=%d\n  batch_size=%d\n'
                '  epochs=%d\n  lr=%.2f\n  memory_size=%d\n  pretrained=%s)'
                % (self.data_root, self.class_ids, len(self.class_ids),
                   self.task_sizes, self.img_size, self.batch_size,
                   self.epochs, self.learning_rate, self.memory_size,
                   self.pretrained))
