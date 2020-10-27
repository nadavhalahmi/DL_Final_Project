#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import Adam, SGD
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
import random
import torch.nn.functional as F
import math
import os
import argparse
from torch.utils import data
from torchvision.models import resnet50, resnet18, resnet34

from attacker.pgd import Linf_PGD
from attacker.pgd import Linf_PGD_weight
from attacker.pgd import Linf_PGD_new
from attacker.pgd import L2_PGD
from attacker.pgd import Linf_PGD_so
from attacker.pgd import Linf_PGD_so_cw


class FixedRandomSampler(data.sampler.Sampler):
    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        print('Training data will be SHUFFLED')
        self.permutation = torch.randperm(len(data_source)).tolist()

    @property
    def num_samples(self):
        return len(self.data_source)

    def resample(self):
        self.permutation = torch.randperm(len(self.data_source)).tolist()
        return self.permutation

    def get_perm(self):
        return self.permutation

    def __iter__(self):
        return iter(self.permutation)

    def __len__(self):
        return len(self.data_source)


class FixedRangeSampler(data.sampler.Sampler):
    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        print('Training data will NOT be shuffle')
        self.permutation = list(range(len(data_source)))

    @property
    def num_samples(self):
        return len(self.data_source)

    def resample(self):
        return self.permutation

    def get_perm(self):
        return self.permutation

    def __iter__(self):
        return iter(self.permutation)

    def __len__(self):
        return len(self.data_source)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--steps', required=True, type=int, help='#adv. steps')
parser.add_argument('--max_norm', required=True, type=float, help='Linf-norm in PGD')
parser.add_argument('--data', required=True, type=str, help='dataset name')
parser.add_argument('--model', required=True, type=str, help='model name')
parser.add_argument('--root', required=True, type=str, help='path to dataset')
parser.add_argument('--model_out', required=True, type=str, help='output path')
parser.add_argument('--resume', action='store', help='Resume training')
parser.add_argument('--train_sampler', action='store', help='Resume training')
parser.add_argument('--resume_from', type=str, help='Resume training')
parser.add_argument('--corruption_prob', '-cprob', type=float, default=0.3, help='The label corruption probability.')
opt = parser.parse_args()

# Data
print('==> Preparing data..')
if opt.data == 'cifar10':
    nclass = 10
    img_width = 32
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root=opt.root, train=True, download=True, transform=transform_train)
    if opt.train_sampler:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,  # TODO: change here to 128
                                                  shuffle=False, num_workers=2, pin_memory=True,
                                                  sampler=FixedRandomSampler(trainset))
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                  shuffle=True, pin_memory=True, num_workers=2)
    train_sampler = trainloader.sampler
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    # statloader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=False, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=opt.root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)     # TODO: change here back to 100
elif opt.data == 'tiny_imagenet':
    nclass = 200
    img_width = 64
    transform_train = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.ImageFolder(os.path.join(opt.root, 'train'), transform=transform_train)
    if opt.train_sampler:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2,
                                                  pin_memory=True, sampler=FixedRandomSampler(trainset))
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    train_sampler = trainloader.sampler

    testset = torchvision.datasets.ImageFolder(os.path.join(opt.root, 'val'), transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
elif opt.data == 'restricted_imagenet':
    data_path = os.path.expandvars(opt.data)
    dataset = DATASETS[opt.data](opt.root)
    train_loader, val_loader = dataset.make_loaders(2, 128, data_aug=False)
    train_sampler = train_loader.sampler
    trainloader = helpers.DataPrefetcher(train_loader)
    testloader = helpers.DataPrefetcher(val_loader)
    nclass = 10
    img_width = 224

else:
    raise NotImplementedError('Invalid dataset')

# Model
if opt.model == 'vgg':
    from models.vgg import VGG

    net = nn.DataParallel(VGG('VGG16', nclass, img_width=img_width).cuda())
elif opt.model == 'aaron':
    from models.aaron import Aaron

    net = nn.DataParallel(Aaron(nclass).cuda())
elif opt.model == 'wide_resnet':
    from models.wideresnet import *

    net = nn.DataParallel(WideResNet(widen_factor=10).cuda())
elif opt.model == 'resnet':
    model_ft = resnet50(pretrained=False, num_classes=10)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 10)
    net = model_ft.cuda()
    net = nn.DataParallel(net)
elif opt.model == 'resnet18':
    model = resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Sequential()
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = torch.nn.Linear(in_features=512, out_features=200, bias=True)
    net = model.cuda()
    net = nn.DataParallel(net)
elif opt.model == 'resnet34':
    model = resnet34()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Sequential()
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = torch.nn.Linear(in_features=512, out_features=200, bias=True)
    net = model.cuda()
    net = nn.DataParallel(net)
elif opt.model == 'resnet50':
    model = resnet50()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Sequential()
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = torch.nn.Linear(in_features=2048, out_features=nclass, bias=True)
    net = model.cuda()
    net = nn.DataParallel(net)
else:
    raise NotImplementedError('Invalid model')

if opt.resume and opt.resume_from:
    print(f'==> Resume from {opt.resume_from}')
    net.load_state_dict(torch.load(opt.resume_from))

best_acc_cw = 0
best_acc_ce = 0
# cudnn.benchmark = True

# Loss function
criterion = nn.CrossEntropyLoss(reduction='none')
test_criterion = nn.CrossEntropyLoss()


def mixup_data(x, y, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def AdaptiveLoss(outputs, targets, dis):
    loss = criterion(outputs, targets)
    factor = torch.log2(2 - 20 * dis)
    # print(loss,factor,dis)
    return torch.mean(factor * loss)


def distance(x_adv, x):
    diff = (x_adv - x).view(x.size(0), -1)
    out = torch.max(torch.abs(diff), 1)[0]
    return out


# Training
def train_natrual(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = test_criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()
    print(f'[TRAIN] Acc: {100. * correct / total:.3f}')


def train_reg(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # print(inputs.max(),inputs.min())
        adv_x = Linf_PGD(inputs, targets, net, opt.steps, opt.max_norm)
        # adv_x = L2_PGD(inputs, targets, net, opt.steps, opt.max_norm)
        dis = distance(adv_x, inputs)
        # print(dis)
        optimizer.zero_grad()
        outputs = net(adv_x)
        loss = test_criterion(outputs, targets)
        loss.backward()
        # print(loss)
        optimizer.step()
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()
    print(f'[TRAIN] Acc: {100. * correct / total:.3f}')


def dirilabel(outputs, targets, eps, already_one_hot=False):
    batch_size, n_class = targets.size(0), 10
    # eps = 0.1
    eps *= 10
    eps = eps.view(-1, 1).cuda()

    if already_one_hot is False:
        one_hot = torch.zeros((batch_size, n_class)).cuda().scatter(1, targets.view(-1, 1), 1)
    else:
        one_hot = targets
    ## here we assume uniform 
    alpha = torch.ones(n_class)
    distri = torch.distributions.Dirichlet(alpha)
    # print(one_hot.shape,eps.shape)
    one_hot_so = one_hot * (1 - eps) + distri.rsample(sample_shape=(batch_size,)).cuda() * eps / n_class
    return one_hot_so, one_hot


def train_soadp(epoch, perm, eps, cw=False, mixup_in_epoch=0, hidden_in_epoch=0, mixup_alpha=0):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    batch_size = 10 # TODO: change here back to 128
    # the indices of the samples for producing adversarial examples

    hidden_index = list(random.sample(range(len(trainloader)), hidden_in_epoch))
    mix_index = list(random.sample(range(len(trainloader)), mixup_in_epoch))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        index = perm[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        correct, total = noraml_adv_train(inputs, targets, index, cw)
        if batch_idx in hidden_index:
            print(f'batch idx {batch_idx} hidden train')
            correct, total = hidden_adv_train(inputs, targets, index, cw)
        if batch_idx in mix_index:
            print(f'batch idx {batch_idx} mixup train')
            correct, total = hidden_mix_adv_train(inputs, targets, index, cw, mixup_alpha)

    print(torch.nonzero(eps).size(0), eps.shape, eps.sum())
    print(f'[TRAIN] Acc: {100. * correct / total:.3f}')


def loss_and_num_corrects(inputs, targets, outputs, index, cw, already_one_hot=False):
    correct = 0
    total = 0
    zero = torch.tensor([0.0]).cuda()
    so_targets, one_hot = dirilabel(inputs, targets, eps[index], already_one_hot)
    outputs = outputs[0]
    if cw:
        real = torch.max(outputs * one_hot - (1 - one_hot) * 100000, dim=1)[0]
        other = torch.max(torch.mul(outputs, (1 - one_hot)) - one_hot * 100000, 1)[0]
        loss1 = torch.max(other - real + 10, zero)
        loss1 = torch.sum(loss1 * eps[index])
        log_prb = F.log_softmax(outputs, dim=1)
        # print(log_prb.shape, y_true.shape)
        loss2 = - (so_targets * log_prb).sum() / inputs.size(0)
        loss = loss1 + loss2
        # loss = torch.sum(loss1)
    else:
        log_prb = F.log_softmax(outputs, dim=1)
        # print(log_prb.shape, y_true.shape)
        loss = - (so_targets * log_prb).sum() / inputs.size(0)
        # loss = test_criterion(outputs, targets)
    loss.backward()
    # print(loss1.item(),loss2.item())
    optimizer.step()
    pred = torch.max(outputs, dim=1)[1]
    if already_one_hot:
        targets = targets.argmax(dim=1)
    correct += torch.sum(pred.eq(targets)).item()
    total += targets.numel()
    return correct, total


def hidden_mix_adv_train(inputs, targets, index, cw, mixup_alpha=1.0):
    """
    training with mixup and then adversarial example in hidden layer
    :param inputs: a batch of samples
    :param targets: predicted labels for the samples
    :param index: the index of the samples in the batch to be taken for producing adversarial example
    :param cw:
    :param mixup_alpha: determines the lambda for mixup
    :return: the correct and the total predictions
    """
    inputs, targets = inputs.cuda(), targets.cuda()
    #with torch.no_grad():
    inputs = net.module.features(inputs)
    inputs = F.avg_pool2d(inputs, 8)
    inputs = inputs.view(-1, 640)
    # convert targets to one_hot
    batch_size, n_class = targets.size(0), 10
    targets = torch.zeros((batch_size, n_class)).cuda().scatter(1, targets.view(-1, 1), 1)
    inputs, targets = mixup_data(inputs, targets, mixup_alpha)
    inputs = inputs.cuda()
    targets = targets.cuda()
    so_targets, one_hot = dirilabel(inputs, targets, eps[index], already_one_hot=True)
    adv_x = Linf_PGD_so_cw(inputs, targets, net.module.classifier, opt.steps, eps[index], one_hot, cw=cw,
                           our=True)
    optimizer.zero_grad()
    eps[index] = distance(adv_x, inputs)
    outputs = net.module.classifier(adv_x)
    return loss_and_num_corrects(inputs, targets, outputs, index, cw, already_one_hot=True)


def hidden_adv_train(inputs, targets, index, cw):
    """
    training with adversarial example in hidden layer
    :param inputs: a batch of samples
    :param targets: predicted labels for the samples
    :param index: the index of the sample in the batch to be taken for producing adversarial example
    :param cw:
    :return: the correct and the total predictions
    """
    inputs, targets = inputs.cuda(), targets.cuda()
    with torch.no_grad():
        inputs = net.module.features(inputs)
        inputs = F.avg_pool2d(inputs, 8)
        inputs = inputs.view(-1, 640)
        inputs = inputs.cuda()
    so_targets, one_hot = dirilabel(inputs, targets, eps[index])
    adv_x = Linf_PGD_so_cw(inputs, so_targets, net.module.classifier, opt.steps, eps[index], one_hot, cw=cw,
                           our=True)
    optimizer.zero_grad()
    eps[index] = distance(adv_x, inputs)
    outputs = net.module.classifier(adv_x)
    return loss_and_num_corrects(inputs, targets, outputs, index, cw)


def noraml_adv_train(inputs, targets, index, cw):
    """
    regular adversarial training
    :param inputs: a batch of samples
    :param targets: predicted labels for the samples
    :param index: the index of the sample in the batch to be taken for producing adversarial example
    :param cw:
    :return: the correct and the total predictions
    """
    inputs, targets = inputs.cuda(), targets.cuda()
    so_targets, one_hot = dirilabel(inputs, targets, eps[index])
    adv_x = Linf_PGD_so_cw(inputs, so_targets, net, opt.steps, eps[index], one_hot, cw=cw, our=False)
    optimizer.zero_grad()
    eps[index] = distance(adv_x, inputs)
    outputs = net(adv_x)
    return loss_and_num_corrects(inputs, targets, outputs, index, cw)


def test_attack(cw):
    correct = 0
    total = 0
    # max_iter = 100
    distortion = 0
    batch = 0
    eps = 0.03
    for it, (x, y) in enumerate(testloader):
        x, y = x.cuda(), y.cuda()
        x_adv = Linf_PGD(x, y, net, 5, eps, cw=cw)
        # pred = torch.max(net(x_adv)[0], dim=1)[1] TODO: unnote this according to whatsapp group
        pred = torch.max(net(x_adv), dim=1)[1]  # TODO: note this according to whatsapp group
        correct += torch.sum(pred.eq(y)).item()
        total += y.numel()
        batch += 1

    correct = str(correct / total)
    # print(f'{distortion/batch},' + ','.join(correct))
    print(f'{eps},' + correct)
    return float(correct)


def test(epoch):
    global best_acc_cw
    global best_acc_ce
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            # loss = criterion(outputs, targets)
            # test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'[TEST] Acc: {100. * correct / total:.3f}')

    # robust_acc_cw = test_attack(True)

    # acc = 100.*correct /total

    if epoch > 60:
        acc = test_attack(True)
        if acc > best_acc_ce:
            print("============ NEW BEST =============")
            print("BEST:", acc)
            best_acc_ce = acc
            model_out = opt.model_out + "best" + f"{mixup_in_epoch}_mix_{hidden_in_epoch}_hidden_{mixup_alpha}_alpha"
            torch.save(net.state_dict(), model_out)
            # if acc> best_acc_ce:
    #     best_acc_ce = acc
    #     model_out = opt.model_out + "best"
    #     torch.save(net.state_dict(), model_out)
    if epoch == 199:
        model_out = opt.model_out + str(epoch)
        torch.save(net.state_dict(), model_out)


if opt.data == 'cifar10':
    epochs = [80, 60, 40, 20]
elif opt.data == 'corrupt_cifar10':
    epochs = [80, 60, 40, 20]
elif opt.data == 'restricted_imagenet':
    epochs = [30, 20, 20, 10]
elif opt.data == 'tiny_imagenet':
    epochs = [30, 20, 20, 10]
elif opt.data == 'stl10':
    epochs = [60, 40, 20]
count = 0
eps = torch.zeros(50000).cuda()


def our_experiment(count, train_perm, eps, cw, mixup_in_epoch, hidden_in_epoch, mixup_alpha):
    train_soadp(count, train_perm, eps, cw=cw, mixup_alpha=mixup_alpha,
                mixup_in_epoch=mixup_in_epoch, hidden_in_epoch=hidden_in_epoch)


mixup_in_epoch = 0
hidden_in_epoch = 0
mixup_alpha = 1.0

print(f'running experiment with mixup_in_epoch={mixup_in_epoch}, '
      f'hidden_in_epoch={hidden_in_epoch}, mixup_alpha={mixup_alpha}')

for epoch in epochs:
    optimizer = SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5.0e-4)
    for it in range(epoch):
        train_perm = train_sampler.get_perm()
        # train_natrual(count)
        # change mixup_in_epoch, hidden_in_epoch and mixup_alpha to control experiments
        our_experiment(count, train_perm, eps, cw=True, mixup_in_epoch=mixup_in_epoch,
                       hidden_in_epoch=hidden_in_epoch, mixup_alpha=mixup_alpha)
        # train_soadp(count, train_perm, eps, cw=True, mixup_alpha=1.0)
        # train_cwadp(count,train_perm,eps, cw=True)
        # train_reg(count)
        test(count)
        count += 1
        train_sampler.resample()
    opt.lr /= 10
