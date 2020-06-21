import glob

i = 0

def make_train_txt(num):
    global i
    paths = glob.glob("cityscapes\\leftImg8bit\\train\\*\\*")

    txt = open("train.txt", "w")

    for path in paths:
        data = path + " " + path.replace("leftImg8bit", "gtFine").replace("gtFine.png", "gtFine_labelTrainIds.png") + "\n"
        txt.write(data)
        i = i + 1
        if i == num:
            break

    i = 0
    txt.close()


def make_test_txt(num):
    global i
    paths = glob.glob("cityscapes\\leftImg8bit\\test\\*\\*")

    txt = open("test.txt", "w")

    for path in paths:
        data = path + " " + path.replace("leftImg8bit", "gtFine").replace("gtFine.png", "gtFine_labelTrainIds.png") + "\n"
        txt.write(data)
        i = i + 1
        if i == num:
            break

    i = 0
    txt.close()


def make_val_txt(num):
    global i
    paths = glob.glob("cityscapes\\leftImg8bit\\val\\*\\*")

    txt = open("val.txt", "w")

    for path in paths:
        data = path + " " + path.replace("leftImg8bit", "gtFine").replace("gtFine.png", "gtFine_labelTrainIds.png") + "\n"
        txt.write(data)
        i = i + 1
        if i == num:
            break

    i = 0
    txt.close()


train_num = 400
test_num = 100
val_num = 100

if True:
    make_train_txt(train_num)
if False:
    make_test_txt(test_num)
if False:
    make_val_txt(val_num)