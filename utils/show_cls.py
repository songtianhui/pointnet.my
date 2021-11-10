from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ModelNetDataset
from pointnet.model import PointNetCls
import torch.nn.functional as F


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')


opt = parser.parse_args()
# print(opt)

# test_dataset = ShapeNetDataset(
#     root='shapenetcore_partanno_segmentation_benchmark_v0',
#     split='test',
#     classification=True,
#     npoints=opt.num_points,
#     data_augmentation=False)

test_dataset = ModelNetDataset(
    root='modelnet40_ply_hdf5_2048',
    split='test_files',
    npoints=opt.num_points,
    data_augmentation=False)

testdataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=True)


classifier = PointNetCls(k=len(test_dataset.classes), feature_transform=True)
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

total_correct = 0
total_testset = 0
for i, data in enumerate(testdataloader, 0):
    points, target = data
    points, target = Variable(points), Variable(target[:, 0])
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    pred, _, _ = classifier(points)
    loss = F.nll_loss(pred, target)

    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    # print('i:%d  loss: %f accuracy: %f' % (i, loss.data.item(), float(correct) / float(points.size()[0])))
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
