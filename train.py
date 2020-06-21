from SegNet import *


def train(SegNet):

    SegNet = SegNet.cuda()
    SegNet.load_weights(PRE_TRAINING)

    train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.SGD(SegNet.parameters(), lr=LR, momentum=MOMENTUM)

    loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(CATE_WEIGHT)).float()).cuda()

    SegNet.train()
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            b_y = b_y.view(BATCH_SIZE, 224, 224)
            output = SegNet(b_x)
            loss = loss_func(output, b_y.long())
            loss = loss.cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 1 == 0:
                print("Epoch:{0} || Step:{1} || Loss:{2}".format(epoch, step, format(loss, ".4f")))

    torch.save(SegNet.state_dict(), WEIGHTS + "SegNet_weights" + str(time.time()) + ".pth")


parser = argparse.ArgumentParser()
parser.add_argument("--class_num", type=int, default=2, help="训练的类别的种类")
parser.add_argument("--epoch", type=int, default=4, help="训练迭代次数")
parser.add_argument("--batch_size", type=int, default=2, help="批训练大小")
parser.add_argument("--learning_rate", type=float, default=0.01, help="学习率大小")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--category_weight", type=float, default=[0.7502381287857225, 1.4990483912788268], help="损失函数中类别的权重")
parser.add_argument("--train_txt", type=str, default="train.txt", help="训练的图片和标签的路径")
parser.add_argument("--pre_training_weight", type=str, default="vgg16_bn-6c64b313.pth", help="编码器预训练权重路径")
parser.add_argument("--weights", type=str, default="./weights/", help="训练好的权重保存路径")
opt = parser.parse_args()
print(opt)

CLASS_NUM = opt.class_num
EPOCH = opt.epoch
BATCH_SIZE = opt.batch_size
LR = opt.learning_rate
MOMENTUM = opt.momentum
CATE_WEIGHT = opt.category_weight
TXT_PATH = opt.train_txt
PRE_TRAINING = opt.pre_training_weight
WEIGHTS = opt.weights


train_data = MyDataset(txt_path=TXT_PATH)

SegNet = SegNet(3, CLASS_NUM)
train(SegNet)
