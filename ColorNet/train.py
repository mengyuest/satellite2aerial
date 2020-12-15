import torch
from torch import nn
import torch.utils.data
from torch.nn.utils import clip_grad_norm_
import torchvision
import numpy as np
from torch.nn.init import normal_, constant_

from PIL import Image
from os.path import join as ospj
import argparse
from datetime import datetime
import time
import os, sys

def get_hyperparameters():
    parser = argparse.ArgumentParser(description="Color Offset Network")
    parser.add_argument('--dataset_path', type=str, default="../../../datasets/satellite_images/")
    parser.add_argument('--train_filelist', type=str, default="../data/satellite_train_colorlist.txt")
    parser.add_argument('--val_filelist', type=str, default="../data/satellite_val_colorlist.txt")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--clip-gradient', '--gd', default=20, type=float, metavar='W', help='gradient norm clipping')
    parser.add_argument('--exp_dir', type=str, default="../../../datasets/satellite_images/")
    parser.add_argument('--exp_name', type=str, default="foobar")
    parser.add_argument('--data_augmentation', action='store_true')
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--test_from', type=str, default=None)
    parser.add_argument('--generated_filelist_name', type=str, default="generated_filelist.txt")
    return parser.parse_args()


class Logger(object):
    def __init__(self):
        self._terminal = sys.stdout
        self._timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")

    def create_log(self, log_path):
        self.log = open(log_path + "/log-%s.txt" % self._timestr, "a", 1)

    def write(self, message):
        self._terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



class Recorder:
    def __init__(self, larger_is_better=True):
        self.history = []
        self.larger_is_better = larger_is_better
        self.best_at = None
        self.best_val = None

    def is_better_than(self, x, y):
        if self.larger_is_better:
            return x > y
        else:
            return x < y

    def update(self, val):
        self.history.append(val)
        if len(self.history) == 1 or self.is_better_than(val, self.best_val):
            self.best_val = val
            self.best_at = len(self.history) - 1

    def is_current_best(self):
        return self.best_at == len(self.history) - 1

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ColorDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, filelist_path, args):
        self.args=args
        self.img_dir=img_dir
        self.filelist=[x.strip() for x in open(filelist_path).readlines()]

        self.input_size = 384
        self.input_mean = np.array([[[0.485, 0.456, 0.406]]])
        self.input_std = np.array([[[0.229, 0.224, 0.225]]])

    def __getitem__(self, item):
        data = self.filelist[item].split( )
        img_path = ospj(self.img_dir, data[0])
        img = np.asarray(Image.open(img_path).resize((self.input_size, self.input_size)).convert('RGB'))/255.0
        img = np.asarray((img - self.input_mean)/self.input_std, dtype=np.float32)
        img = torch.tensor(img).permute(2,0,1)
        label = torch.tensor([float(data[1]), float(data[2]), float(data[3])])
        return img,  data[0], label

    def __len__(self):
        return len(self.filelist)


class ColorNet(nn.Module):
    def __init__(self, args):
        super(ColorNet, self).__init__()
        self.args = args
        self.base_model = getattr(torchvision.models, "resnet18")(True)

        feature_dim = getattr(self.base_model, "fc").in_features
        setattr(self.base_model, "fc", nn.Dropout(p=self.args.dropout))
        self.new_fc = nn.Linear(feature_dim, 3)
        normal_(self.new_fc.weight, 0, 0.001)
        constant_(self.new_fc.bias, 0)

    def forward(self, x):
        feat = self.base_model(x)
        logits = self.new_fc(feat)
        return logits

def get_loaders(args):
    # split train/val split
    train_loader = torch.utils.data.DataLoader(
        ColorDataset(args.dataset_path, args.train_filelist, args),
        batch_size=args.batch_size,
        shuffle=True if args.test_from is None else False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        ColorDataset(args.dataset_path, args.val_filelist, args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    return train_loader, val_loader

def cal_loss(pred, target_var):
    return torch.mean((pred - target_var) ** 2)

def train(epoch_i, train_loader, model, optimizer, args):
    model.train()
    losses = AverageMeter()
    for i, input_tuple in enumerate(train_loader):
        t0 = time.time()
        target = input_tuple[-1].cuda()
        target_var = torch.autograd.Variable(target)
        input_var = torch.autograd.Variable(input_tuple[0])

        pred = model(input_var)

        loss = cal_loss(pred, target_var)

        loss.backward()
        losses.update(loss.detach().cpu().item())

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        t1 = time.time()

        if i % args.print_freq == 0:
            print_output = ('Epoch:[{0:02d}][{1:03d}/{2:03d}] '
                            'Time {batch_time:.3f}\t'
                            'Loss {loss.val:.6f} ({loss.avg:.6f})'.format(
                epoch_i, i, len(train_loader), batch_time=t1 - t0, loss=losses))
            print(print_output)


def validate(val_loader, model, args, test=False, split="train"):
    model.eval()
    val_losses = AverageMeter()
    all_preds = []
    all_targets = []
    if test:
        writer=open("../data/%s"%(args.generated_filelist_name.replace(".txt", "_%s.txt"%split)), "w")
    with torch.no_grad():
        for i, input_tuple in enumerate(val_loader):
            t0 = time.time()
            target = input_tuple[-1].cuda()
            target_var = torch.autograd.Variable(target)
            input_var = torch.autograd.Variable(input_tuple[0])

            pred = model(input_var)

            if i == 0:
                print(pred[0].detach().cpu().numpy())

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

            loss = cal_loss(pred, target_var)

            val_losses.update(loss.cpu().item())
            t1 = time.time()

            if test:
                fnames = input_tuple[1]
                for j in range(len(fnames)):
                    writer.writelines("%s %.7f %.7f %.7f\n"%(fnames[j], pred.cpu()[j, 0], pred.cpu()[j, 1], pred.cpu()[j, 2]))

            if i % args.print_freq == 0:
                print_output = ('Test:[{0:03d}/{1:03d}] '
                                'Time {batch_time:.3f}\t'
                                'Loss {loss.val:.6f} ({loss.avg:.6f})'.
                                format(i, len(val_loader), batch_time=t1 - t0, loss=val_losses))
                print(print_output)

    all_preds = torch.cat(all_preds, 0)
    all_targets = torch.cat(all_targets, 0)

    all_loss = cal_loss(all_preds, all_targets)

    return all_loss.item()


def main():
    args = get_hyperparameters()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    logger = Logger()
    full_exp_path = ospj(args.exp_dir, "g" + logger._timestr + "color_" + args.exp_name)
    os.makedirs(full_exp_path, exist_ok=True)
    saved_model_path = ospj(full_exp_path, "models")
    os.makedirs(saved_model_path, exist_ok=True)
    logger.create_log(full_exp_path)
    sys.stdout = logger

    logger.log.write("python " + " ".join(sys.argv)+"\n")

    # model
    model = ColorNet(args)
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.test_from is not None:
        assert len(args.gpus) <= 1 # only use 1 gpu for writing to filelist
        print('LOAD test model from %s' % (args.test_from))
        model_dict = model.state_dict()
        saved_model = torch.load(args.test_from)

        saved_model['state_dict'] = {k.replace("module.", ""): saved_model["state_dict"][k].cpu()
                                     for k in saved_model["state_dict"]}

        model_dict.update(saved_model["state_dict"])
        model.load_state_dict(model_dict)



    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    train_loader, val_loader = get_loaders(args)

    val_loss_record = Recorder(larger_is_better=False)

    if args.test_from is not None:
        val_train_loss = validate(train_loader, model, args, test=True, split="train")
        val_val_loss = validate(val_loader, model, args, test=True, split="val")
        print("Test-(train split)  overall loss: %.6f" % (val_train_loss))
        print("Test-(val   split)  overall loss: %.6f" % (val_val_loss))
    else:
        # MAIN LOOP
        for epoch_i in range(args.epochs):
            train(epoch_i, train_loader, model, optimizer, args)
            # val
            if (epoch_i + 1) % args.eval_freq == 0 or epoch_i == args.epochs - 1:
                val_loss = validate(val_loader, model, args)
                val_loss_record.update(val_loss)

                print("Val-%d  overall loss: %.6f (best: %.6f at %d)"%(epoch_i, val_loss, val_loss_record.best_val, val_loss_record.best_at))
                # if the overall loss very small, save the model
                if val_loss_record.is_current_best():
                    torch.save({'state_dict': model.state_dict()}, saved_model_path+'/color.pth.tar')


if __name__ == "__main__":
    main()
