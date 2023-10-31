import torch
from torch.utils.data import Dataset, DataLoader
import configargparse
import cv2
import numpy as np
import utils
import pandas as pd
import os
import tqdm
import matplotlib.pyplot as plt
import random

'''
python train.py --epochs=40 --lr=0.001 --i_save=4 --batch_size=16 --gamma=0.5 -freeze
loss能下降 准确率能上升
0 python train.py --epochs=80 --lr=0.0005 --i_save=10 --batch_size=16 --gamma=0.8 -freeze
python train.py --epochs=100 --lr=0.001 --i_save=10 --batch_size=16 --gamma=0.5  --model_name resnet_18 不加数据增强seed=0能达到0.66,seed=3407 0.58
'''


def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--image_path', default='./data/image/')
    parser.add_argument('--label_path', default='./data/')
    parser.add_argument('--save_path', default='./output/')
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--model_name', default='resnet_101')

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--i_save', type=int, default=1)
    parser.add_argument('--resolution', default=512)
    parser.add_argument('--momentum', default=0.1)

    parser.add_argument('--freeze_model', '-freeze',
                        default=False, action='store_true')
    parser.add_argument('--freeze_layers', type=int,default=0)
    parser.add_argument('--step_size', default=5)
    parser.add_argument('--gamma', type=float, default=0.5)

    return parser


torch.manual_seed(0)
torch.cuda.manual_seed(0)

def crop(img:np.ndarray):
    mask=np.random.rand(20,20)>0.05
    mask=cv2.resize(mask.astype('uint8'),(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
    for i in range(3):
        img[:,:,i]=img[:,:,i]*mask
    return img

# augment_fun = [lambda x:np.transpose(x, (0, 2, 1)), lambda x: x[::-1], lambda x:x[:, ::-1],
#                lambda x: x, lambda x: (x+np.random.randn(*x.shape)/10).astype('float32'),crop]

augment_fun = [lambda x:np.transpose(x, (0, 2, 1)), lambda x: x[::-1], lambda x:x[:, ::-1],crop]

def data_augment(image: np.ndarray):
    rand = random.random()
    return augment_fun[int(rand*len(augment_fun))](image)

def data_augment(image: np.ndarray):
    for fun in augment_fun:
        rand = random.random()
        if rand > 0.9:
            image = fun(image)
    return image 


def get_image(image_path: str, args):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (args.resolution, args.resolution))
    image = image.astype('float32')
    image = image/255
    image = np.transpose(image, (2, 0, 1))
    return image


class dataset(Dataset):
    def __init__(self, args):
        super(dataset, self).__init__()
        self.args = args
        self.path = args.image_path
        self.image_path = []
        self.label = []
        with open(args.label_path+'train.txt', 'r') as f:
            for line in f.readlines():
                self.image_path.append(self.path+line.split('\t')[0])
                self.label.append(int(line.split('\t')[1]))

    def __getitem__(self, index):
        image = get_image(self.image_path[index], self.args)
        image = data_augment(image)
        image = torch.tensor(image.copy()).to(self.args.device)
        label = self.label[index]
        return image, label

    def __len__(self):
        return len(self.label)


def test(args, model, txt_name='test.txt'):
    path = args.image_path
    image_path = []
    label = []
    with open(args.label_path+txt_name, 'r') as f:
        for line in f.readlines():
            image_path.append(path+line.split('\t')[0])
            label.append(int(line.split('\t')[1]))
    n = 0
    with torch.no_grad():
        for i in range(len(image_path)):
            image = get_image(image_path[i], args)
            image = torch.tensor(image).to(args.device)
            image = torch.unsqueeze(image, 0)
            out = model(image)
            out = out.argmax()
            if int(out) == label[i]:
                n += 1
    return n/len(label)


def main():
    parser = config_parser()
    args = parser.parse_args()
    dataloader = DataLoader(
        dataset(args), batch_size=args.batch_size, shuffle=True)
    model = utils.model[args.model_name]().to(args.device)
    if args.freeze_model:
        print('freeze backbone')
        model.freeze(args.freeze_layers)
    model.train()

    loss_fun = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=args.lr, momentum=args.momentum)
    lr_shedular = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)
    train_loss = []
    test_acc = []
    train_acc = []
    for epoch in tqdm.tqdm(range(args.epochs)):
        for i, (image, label) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(image)
            loss = loss_fun(output, label.to(args.device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(float(loss))
            lr_shedular.step()
        if (epoch+1) % args.i_save == 0:
            model.eval()
            test_acc.append(test(args, model, 'test.txt'))
            train_acc.append(test(args, model, txt_name='train.txt'))
            save_path = args.save_path+'exp'+str(epoch)+'/'
            try:
                os.mkdir(save_path)
            except:
                pass
            pd.DataFrame({'train acc': np.array(
                train_acc), 'test acc': test_acc}).to_csv(save_path+'output.csv')
            plt.plot(train_loss)
            plt.savefig(save_path+'loss.pdf')
            torch.save(model, save_path+'model.pt')
            model.train()


if __name__ == '__main__':
    main()
