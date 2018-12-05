import numpy as np
from torchvision import datasets
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


class JiaGuWenDataSet(Dataset):

    def __init__(self, dir, train=True, transform=None):
        self.dir = dir
        self.train = train
        self.transform = transform

        self.train_file = os.path.join(self.dir, 'train.pt')
        self.test_file = os.path.join(self.dir, 'test.pt')

        if train:
            if not self._check_exists(self.train_file):
                self._generate_data_pt()
            self.train_data, self.train_labels = torch.load(self.train_file)
        else:
            if not self._check_exists(self.test_file):
                self._generate_data_pt()
            self.test_data, self.test_labels = torch.load(self.test_file)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self, path):
        return os.path.exists(path)

    def _generate_data_pt(self):
        print("starting to load {} data...")
        if self.train:
            data_dir = os.path.join(self.dir, 'train')
            save_dir = self.train_file
            imageset = datasets.ImageFolder(data_dir, self.transform)
        else:
            dir = os.path.join(self.dir, 'test')
            save_dir = self.test_file
            imageset = datasets.ImageFolder(dir, self.transform)

        data = torch.stack([imageset[i][0] for i in range(0, len(imageset))], 0)
        label = torch.Tensor([imageset[i][1] for i in range(0, len(imageset))], 0)

        obj = data, label
        torch.save(obj, save_dir)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        if dataset.train:
            self.labels = dataset.train_labels
        else:
            self.labels = dataset.test_labels
        # labels中的去重类别
        self.labels_set = list(set(self.labels.numpy()))
        # 将所有样本对label进行聚合
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        # 对每个类中的样本进行shuffle
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])

        # 对每个Lable进行计数
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        # 总的类别数
        self.n_classes = n_classes
        # 每类的样本数
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0

        while self.count + self.batch_size < len(self.dataset):
            # 在lables_set中不重复地随机抽取n_class个样本
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            # 对每一个类
            for class_ in classes:
                # 对每个类，都取出25个样本的索引追加到indices中，因有10个类，每个类25个样本，总共有250个样本索引
                indices.extend(self.label_to_indices[class_]
                              [self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])

                self.used_label_indices_count[class_] += self.n_samples
                # 如果某个类别用完了数据，就打乱后重新开始取
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            # 返回250个样本的索引
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
