import torch
import numpy as np
from tensorboardX import SummaryWriter


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, conf=None):

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        print('----------------Epoch: {}/{}----------------'.format(epoch + 1, n_epochs))
        writer = SummaryWriter(conf.writer_path)
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        writer.add_scalar('train/Average loss', train_loss, epoch+1)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            writer.add_scalar('train/' + metric.name(), metric.value(), epoch+1)

        # Test stage
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,val_loss)
        writer.add_scalar('test/Average loss', val_loss, epoch+1)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            writer.add_scalar('test/' + metric.name(), metric.value(), epoch+1)

        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    # 重设metric
    for metric in metrics:
        metric.reset()

    # 设置模型为train模式
    model.train()

    # 初始化loss值
    losses = []
    total_loss = 0

    # 遍历每一批数据进行训练
    for batch_idx, (data, target) in enumerate(train_loader):

        # 将数据放到指定GPU上
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        # 梯度归0
        optimizer.zero_grad()

        # 前向计算
        outputs = model(*data)
        # print(outputs.size())

        # 计算损失
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

        # 追加每次损失
        losses.append(loss.item())
        # 汇总每次损失的和
        total_loss += loss.item()

        # 反向计算
        loss.backward()
        optimizer.step()

        # 计算评估指标
        for metric in metrics:
            metric(outputs, target, loss_outputs)

        # 获取信息
        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    # 返回平均损失
    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):

    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
