import torch
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import fit
from datasets import JiaGuWenDataSet, BalancedBatchSampler
from networks import EmbeddingNet
from losses import OnlineTripletLoss
from utils import RandomNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric
import argparse
from config import get_config


def run_training(conf):
    # 指定GPU
    # device = torch.device(conf.cuda_id if torch.cuda.is_available() else "cpu")
    cuda = torch.cuda.is_available()

    # 1.数据加载
    # 参数定义
    data_dir = conf.data_dir
    train_n_classes = conf.train_n_classes
    test_n_classes = conf.test_n_classes
    train_n_samples = conf.train_n_samples
    test_n_samples = conf.test_n_samples

    # 定义图像的转换，
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机截取224*224大小的PIL图像
        transforms.RandomHorizontalFlip(),  # 以0.5的概率水平翻转给定的PIL图像
        transforms.ToTensor(),
        # convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in range [0.0,1.0]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
        transforms.Resize(256),  # resize
        transforms.CenterCrop(224),  # 在图片的中间区域进行裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 获取数据集
    train_dataset = JiaGuWenDataSet(dir=data_dir, train=True, transform=train_transform)
    test_dataset = JiaGuWenDataSet(dir=data_dir, train=False, transform=test_transform)
    print('data size:{}, label size:{}'.format(train_dataset.train_data.size(), test_dataset.test_data.size()))

    # 采样类：用于triplet on line采样
    train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=train_n_classes, n_samples=train_n_samples)
    test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=test_n_classes, n_samples=test_n_samples)

    # 创建DataLoader
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

    # 2.构建网络
    # 参数定义
    margin = conf.margin
    lr = conf.lr
    weight_decay = conf.weight_decay
    step_size = conf.step_size
    gamma = conf.gamma
    last_epoch = conf.last_epoch
    n_epochs = conf.n_epochs
    log_interval = conf.log_interval

    # 模型（只需要embedding部分，即将图片转换成vector）
    embedding_net = EmbeddingNet()
    model = embedding_net
    if cuda:
        model.cuda()

    # 损失
    loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))

    # 优化函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 学习率衰减
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)

    # 3.训练
    fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
        metrics=[AverageNonzeroTripletsMetric()], conf=conf)

    # 4.保存
    save_path = conf.save_path
    torch.save(model.state_dict, save_path)


# nohup python run.py --save_path ./model/model_test01.pt --writer_path ./log/board_test01  >log/log_test-1.log &
# tensorboard --logdir ./log/tensorbord --port=6006
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for image classifier')
    parser.add_argument("--save_path", default='./model/model.pt', type=str)
    parser.add_argument("--writer_path", default='./log/tensorbord', type=str)

    args = parser.parse_args()

    conf = get_config()
    conf.save_path = args.save_path
    conf.writer_path = args.writer_path

    run_training(conf)

