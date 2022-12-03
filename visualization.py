
# 四川大学，跨模态行人重识别检索可视化代码
# liu qiang

from __future__ import print_function
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
import time

import scipy.io as scio
import matplotlib.pyplot as plt
from PIL import Image


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline')

parser.add_argument('--resume', '-r', default='sysu_agw_p4_n5_lr_0.1_seed_0_best65.34.t',
                    type=str, help='resume from checkpoint')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')

parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',   help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=2048, type=int,   metavar='D', help='feature dimension')

parser.add_argument('--img_w', default=144, type=int,  metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,   metavar='imgh', help='img height')

parser.add_argument('--batch-size', default=32, type=int,  metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,  metavar='tb', help='testing batch size')
parser.add_argument('--method', default='id', type=str,  metavar='m', help='Method type')
parser.add_argument('--drop', default=0.0, type=float,  metavar='drop', help='dropout ratio')
parser.add_argument('--trial', default=1, type=int,   metavar='t', help='trial')

parser.add_argument('--gpu', default='5', type=str,  help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(1)
dataset = args.dataset
if dataset == 'sysu':
    data_path = '/home/bobo/E/liuq/data/SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]

if dataset == 'regdb':
    data_path = '/home/dl/VSST/liuq/data/RegDB/'
    n_class = 206
    test_mode = [1, 2]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Building model..')
net = embed_net(n_class, arch=args.arch)
net.to(device)
cudnn.benchmark = True

print('==> Resuming from checkpoint..')
checkpoint_path = args.model_path
if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

if args.method == 'id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])
end = time.time()

query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

nquery = len(query_label)
ngall = len(gall_label)
print("Dataset statistics:")
print("  ------------------------------")
print("  subset   | # ids | # images")
print("  ------------------------------")
print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
print("  ------------------------------")

queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

feature_dim = args.low_dim

if args.arch == 'resnet50':
    pool_dim = 2048
elif args.arch == 'resnet18':
    pool_dim = 512

def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, feature_dim))
    gall_feat_pool = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            pool_feat, feat = net(input, input, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_pool[ptr:ptr + batch_num, :] = pool_feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat, gall_feat_pool


def extract_query_feat(query_loader):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, feature_dim))
    query_feat_pool = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            pool_feat, feat = net(input, input, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_pool[ptr:ptr + batch_num, :] = pool_feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat, query_feat_pool


def cal_similarity(query_feat, gallery_feat, q_pids, g_pids, q_camids, g_camids):
    """calculate the similarty of the query and gallery
    Args:
        input features,labels and camera ids

    Returns:
        for each query image,return 10 indexs of similar images in gallery

    """
    # The larger the cosine distance, the more similar it is
    distmat = -np.matmul(query_feat, np.transpose(gallery_feat))
    num_q = query_feat.shape[0]
    num_g = gallery_feat.shape[0]
    max_rank = 10
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (pred_label == q_pids[:, np.newaxis]).astype(np.int32)

    return_matchs = []
    return_index = []

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)

        new_matchs = matches[q_idx][keep]
        new_index = indices[q_idx][keep]

        return_matchs.append(new_matchs[:10])
        return_index.append(new_index[:10])

    return np.asarray(return_index).astype(np.int32), np.asarray(return_matchs).astype(np.int32)


def save_visualfig(query_imgpath,gall_imgpath,index,match):
    """
     :param query_imgpath: the path of query images,type:list
     :param gall_imgpath:  the path of gallery images,type:list
     :param index:the index for the similar imgs of querys
     :param match: 1 or 0，type list
     :return: none

    """
    bestmatch_id =np.argsort(-match[:,0:30].sum(1))   # 负号代表是正排序

    for k in range(0,1000):
        idx =bestmatch_id[k]

        print("{}-th image,Top 10 images!,the idx is {}".format(k,idx))

        q_path =query_imgpath[int(idx)]

        try:
            fig = plt.figure(figsize=(12,2))
            ax =fig.add_subplot(1,11,1)
            ax.axis('off')

            q_img =Image.open(q_path)

            #q_img =plt.imread(q_path)
            ax.imshow(q_img.resize((144,288),Image.ANTIALIAS))   # 远程取消不显示，否则提示Xmanager！
            ax.set_title("query")
            #ax.set_title(q_path.split(".")[0].split("ori_data/")[-1].replace("/","_"))
            for i in range(10):
                ax =fig.add_subplot(1, 11, i + 2)
                ax.axis('off')
                img_path =gall_imgpath[index[idx][i]]

                #im = plt.imread(img_path)
                im =Image.open(img_path)
                ax.imshow(im.resize((144,288),Image.ANTIALIAS))  # 远程取消不显示，否则提示Xmanager！
                #plt.pause(0.001)
                if match[idx][i]:
                    ax.set_title('%d' % (i + 1), color='green')
                else:
                    ax.set_title('%d' % (i + 1), color='red')

        except RuntimeError:
            for i in range(10):
                img_path = gall_img[index[0][i]]
                print(img_path[0])
                print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

        vasual_save_path ="visual2_saved2/"     # 设定图片保存路径
        if not os.path.exists(vasual_save_path):
           os.mkdir(vasual_save_path)
        fig.savefig(vasual_save_path+"query_{}.png".format((str(k)+"_"+q_path.split(".")[0].split("ori_data/")[-1])).replace("/","_"))
        plt.close()

query_feat, query_feat_pool = extract_query_feat(query_loader)
gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
gall_feat, gall_feat_pool = extract_gall_feat(trial_gall_loader)

# fc feature
distmat = np.matmul(query_feat, np.transpose(gall_feat))
index,match =cal_similarity(query_feat,gall_feat,query_label,gall_label,query_cam,gall_cam)
save_visualfig(query_img,gall_img,index,match)


