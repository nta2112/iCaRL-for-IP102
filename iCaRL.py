import copy
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from config import Config
from iIP102 import iIP102
from myNetwork import network

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(target.device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


class iCaRLmodel:

    def __init__(self, numclass, feature_extractor, batch_size, task_size,
                 memory_size, epochs, learning_rate, device, config=None):
        super(iCaRLmodel, self).__init__()
        if config is None:
            config = Config()
        self.config = config
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.model = network(numclass, feature_extractor)
        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = numclass
        self.task_size = task_size
        self.batchsize = batch_size
        self.memory_size = memory_size

        self.norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        img_size = config.img_size
        resize_size = config.cache_size

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            self.norm])

        self.train_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomCrop((img_size, img_size), padding=8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25),
            transforms.ToTensor(),
            self.norm])

        self.test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            self.norm])

        self.classify_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=1.),
            transforms.ToTensor(),
            self.norm])

        self.train_dataset = iIP102(config.data_root, transform=self.train_transform,
                                    class_ids=config.class_ids,
                                    cache_size=config.cache_size)
        self.test_dataset = iIP102(config.data_root, test_transform=self.test_transform,
                                   class_ids=config.class_ids,
                                   cache_size=config.cache_size)

        self.train_loader = None
        self.test_loader = None
        self.old_model = None
        self.save_dir = config.output_dir

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
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=False,
                                 batch_size=self.batchsize)
        return train_loader, test_loader

    def train(self):
        accuracy = 0
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate,
                        momentum=0.9, nesterov=True, weight_decay=0.00001)
        milestones = [e for e in (int(self.epochs * f) for f in self.config.lr_fractions) if e > 0]
        for epoch in range(self.epochs):
            if epoch in milestones:
                idx = milestones.index(epoch)
                new_lr = self.learning_rate * self.config.lr_multipliers[idx]
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

        self.old_model = copy.deepcopy(self.model).to(self.device)
        self.old_model.eval()

        os.makedirs(self.save_dir, exist_ok=True)
        filename = os.path.join(
            self.save_dir,
            'icarl_ip102_task%d_acc%.3f_knn%.3f.pkl' % (task_id, accuracy, KNN_accuracy))
        torch.save({'state_dict': self.model.state_dict(),
                    'numclass': self.numclass,
                    'class_ids': self.config.class_ids}, filename)
        print('model saved to %s' % filename)
        self._log_result(task_id, accuracy, KNN_accuracy)

    def _log_result(self, task_id, accuracy, knn):
        path = os.path.join(self.save_dir, 'results.csv')
        header = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if header:
                w.writerow(['task', 'numclass', 'softmax_acc', 'knn_acc'])
            w.writerow([task_id, self.numclass,
                        round(float(accuracy), 3), round(float(knn), 3)])

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
