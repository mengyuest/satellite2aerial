import os, sys
from os.path import join as ospj
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
import kornia
import argparse
from logger import Logger

class PairedImageDataset(data.Dataset):
    def __init__(self, lr_img_path, lr_filelist_path, hr_img_path, hr_filelist_path, args):
        self.args=args
        self.lr_img_path = lr_img_path
        self.hr_img_path = hr_img_path
        self.lr_filelist_path = lr_filelist_path
        self.hr_filelist_path = hr_filelist_path

        self.lr_img_list = [x.strip() for x in open(self.lr_filelist_path).readlines()]
        self.hr_img_list = [x.strip() for x in open(self.hr_filelist_path).readlines()]

        # -85.61112_30.197733_28cm.tif -> -85.61112_30.197733_50cm.png
        self.paired_lr_img_list = [x.replace("28cm.tif", "50cm.png") for x in self.hr_img_list]



    def __getitem__(self, item):
        lr_img_name = self.paired_lr_img_list[item]
        hr_img_name = self.hr_img_list[item]

        lr_img = Image.open(ospj(self.lr_img_path, lr_img_name)).convert('RGB')
        hr_img = Image.open(ospj(self.hr_img_path, hr_img_name)).convert('RGB')

        lr_img = np.asarray(lr_img) / 255.0
        hr_img = np.asarray(hr_img) / 255.0

        lr_img = kornia.image_to_tensor(lr_img).squeeze()
        hr_img = kornia.image_to_tensor(hr_img).squeeze()

        return lr_img, hr_img

    def __len__(self):
        return len(self.hr_img_list)

class TVDenoise(torch.nn.Module):
    def __init__(self, args):
        super(TVDenoise, self).__init__()
        self.l2_term = torch.nn.MSELoss(reduction='mean')
        self.l1_term = torch.nn.L1Loss(reduction='mean')
        self.psnr = kornia.losses.PSNRLoss(max_val=1.0)
        self.ssim=kornia.losses.SSIM(5, reduction='mean')
        self.regularization_term = kornia.losses.TotalVariation()
        self.args=args

        self.xyxy = torch.nn.Parameter(data=torch.tensor([[0.], [0.], [713], [713]]), requires_grad=True)
        self.mem = torch.nn.Parameter(data=torch.tensor(
            [[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0],
             [0, 0, 0, 1]]), requires_grad=False)

    def forward(self, src_img, dst_img):

        new_image = self.get_new_image(src_img, dst_img)
        return new_image

    def get_new_image(self, src_img, dst_img):
        self.boxes=torch.matmul(self.mem, self.xyxy).reshape(1, 4, 2)
        return kornia.crop_and_resize((src_img), self.boxes, dst_img.shape[-2:])

def train(epoch_i, data_loader, network, optimizer, args):
    num_iters = len(data_loader)
    loss_list=[]
    l1loss_list=[]
    l2loss_list=[]
    for i, input_tuple in enumerate(data_loader):
        optimizer.zero_grad()
        lr_img, hr_img = input_tuple
        resized_img = network(hr_img, lr_img)

        l1loss = network.l1_term(resized_img, lr_img)
        l2loss = network.l2_term(resized_img, lr_img)

        if args.use_l2_loss:
            loss = l2loss
        else:
            loss = l1loss

        loss.backward()
        optimizer.step()

        loss_list.append(loss.detach().numpy())
        l1loss_list.append(l1loss.item())
        l2loss_list.append(l2loss.item())
        if i % 20 == 0:
            print("[{:2d}] [{:3d}/{:3d}]: loss {:.5f} l1:{:.5f} l2:{:.5f}".
                  format(epoch_i, i, num_iters, loss.item(), l1loss.item(), l2loss.item()),
                  "crop", network.xyxy.detach().numpy().flatten())

    print("Averge loss: %.5f\tl1: %.5f\tl2: %.5f"%(np.mean(loss_list), np.mean(l1loss_list), np.mean(l2loss_list)))

def main():
    parser = argparse.ArgumentParser(description="Learnable Cropping Images")
    parser.add_argument('--use_l2_loss', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--not_pin_memory', action='store_true')
    parser.add_argument('--lr_img_path', type=str, default="../../dataset/satellite_images/")
    parser.add_argument('--lr_filelist_path', type=str, default="data/satellite_images_filelist.txt")
    parser.add_argument('--hr_img_path', type=str, default="../../dataset/aerial_images/")
    parser.add_argument('--hr_filelist_path', type=str, default="data/aerial_images_filelist.txt")
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=100.0)
    parser.add_argument('--exp_name', type=str, default="learncrop")

    args = parser.parse_args()

    logger = Logger()
    exp_dir = ospj("exps", args.exp_name+logger._timestr)
    os.makedirs(exp_dir, exist_ok=True)

    logger.create_log(exp_dir)
    sys.stdout = logger
    if args.use_l2_loss:
        print("use l2 loss")
    else:
        print("use l1 loss")

    dataset = PairedImageDataset(args.lr_img_path, args.lr_filelist_path, args.hr_img_path, args.hr_filelist_path, args)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                                              num_workers=args.workers, pin_memory=not args.not_pin_memory, sampler=None,
                                              drop_last=False)
    network = TVDenoise(args)
    optimizer = torch.optim.SGD(network.parameters(), lr=args.learning_rate, momentum=0.9)

    for epoch_i in range(args.num_epochs):
        train(epoch_i, data_loader, network, optimizer, args)


if __name__ == "__main__":
    main()