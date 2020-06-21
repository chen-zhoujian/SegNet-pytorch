from SegNet import *


def test(SegNet):

    SegNet.load_state_dict(torch.load(WEIGHTS))
    SegNet.eval()

    paths = os.listdir(SAMPLES)

    for path in paths:

        image_src = cv.imread(SAMPLES + path)
        image = cv.resize(image_src, (224, 224))

        image = image / 255.0
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        image = torch.unsqueeze(image, dim=0)

        output = SegNet(image)
        output = torch.squeeze(output)
        output = output.argmax(dim=0)
        output_np = cv.resize(np.uint8(output), (2048, 1024))

        image_seg = np.zeros((1024, 2048, 3))
        image_seg = np.uint8(image_seg)

        colors = COLORS

        for c in range(CLASS_NUM):
            image_seg[:, :, 0] += np.uint8((output_np == c)) * np.uint8(colors[c][0])
            image_seg[:, :, 1] += np.uint8((output_np == c)) * np.uint8(colors[c][1])
            image_seg[:, :, 2] += np.uint8((output_np == c)) * np.uint8(colors[c][2])

        image_seg = Image.fromarray(np.uint8(image_seg))
        old_image = Image.fromarray(np.uint8(image_src))

        image = Image.blend(old_image, image_seg, 0.3)

        # 将背景或空类去掉
        image_np = np.array(image)
        image_np[output_np == 0] = image_src[output_np == 0]
        image = Image.fromarray(image_np)
        image.save(OUTPUTS + path)

        print(path + " is done!")


parser = argparse.ArgumentParser()
parser.add_argument("--class_num", type=int, default=2, help="预测的类别的种类")
parser.add_argument("--weights", type=str, default="weights/SegNet_weights1592624668.4279704.pth", help="训练好的权重路径")
parser.add_argument("--colors", type=int, default=[[0, 0, 0], [0, 255, 0]], help="类别覆盖的颜色")
parser.add_argument("--samples", type=str, default="samples//", help="用于测试的图片文件夹的路径")
parser.add_argument("--outputs", type=str, default="outputs//", help="保存结果的文件夹的路径")
opt = parser.parse_args()
print(opt)

CLASS_NUM = opt.class_num
WEIGHTS = opt.weights
COLORS = opt.colors
SAMPLES = opt.samples
OUTPUTS = opt.outputs


SegNet = SegNet(3, CLASS_NUM)
test(SegNet)
