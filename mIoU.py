from SegNet import *


parser = argparse.ArgumentParser()
parser.add_argument("--class_num", type=int, default=2, help="预测的类别的种类")
parser.add_argument("--weights", type=str, default="weights/SegNet_weights1592624668.4279704.pth", help="训练好的权重路径")
parser.add_argument("--val_paths", type=str, default="val.txt", help="验证集的图片和标签的路径")
opt = parser.parse_args()
print(opt)

CLASS_NUM = opt.class_num
WEIGHTS = opt.weights
VAL_PATHS = opt.val_paths


SegNet = SegNet(3, CLASS_NUM)
SegNet.load_state_dict(torch.load(WEIGHTS))
SegNet.eval()

paths = open(VAL_PATHS, "r")
mIoU = []
for index, line in enumerate(paths):
    line.rstrip("\n")
    line.lstrip("\n")
    path = line.split()

    image = cv.imread(path[0])
    image = cv.resize(image, (224, 224))
    image = image / 255.0  # 归一化输入
    image = torch.Tensor(image)
    image = image.permute(2, 0, 1)  # 将图片的维度转换成网络输入的维度（channel, width, height）
    image = torch.unsqueeze(image, dim=0)

    output = SegNet(image)
    output = torch.squeeze(output)
    output = output.argmax(dim=0)
    predict = cv.resize(np.uint8(output), (2048, 1024))

    label = cv.imread(path[1])
    target = label[:, :, 0]

    # 自己写的方法
    intersection = []
    union = []
    iou = 0
    for i in range(1, CLASS_NUM):
        intersection.append(np.sum(predict[target == i] == i))
        union.append(np.sum(predict == i) + np.sum(target == i) - intersection[i-1])
        iou += intersection[i-1]/union[i-1]

    # 用numpy库实现的方法
    # intersection = np.logical_and(target, predict)
    # union = np.logical_or(target, predict)
    # iou = np.sum(intersection) / np.sum(union)

    mIoU.append(iou/CLASS_NUM)
    print("miou_{0}:{1}".format(index, format(mIoU[index], ".4f")))

paths.close()

file = open("result.txt", "a")

print("\n")
print("mIoU:{}".format(format(np.mean(mIoU), ".4f")))

file.write("评价日期：" + str(time.asctime(time.localtime(time.time()))) + "\n")
file.write("使用的权重：" + WEIGHTS + "\n")
file.write("mIoU: " + str(format(np.mean(mIoU), ".4f")) + "\n")

file.close()