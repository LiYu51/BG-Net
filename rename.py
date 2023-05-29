import os


def eachFile(filepath):
    pathDir = os.listdir(filepath)
    file_list = []
    for path in pathDir:
        file_list.append(path.split('.')[0])
    return file_list


if __name__ == '__main__':

    pic_path = 'F:/yl/project/pytorch_nested_unet_master/inputs/dsb2018_96/masks/0/'
    pic_list = eachFile(pic_path)

    print(len(pic_list))
    pic_set = set(pic_list)
    print(pic_set)

    for imgname in pic_set:
        imgname = pic_path + imgname
        if os.path.exists(imgname + '.png'):
            os.rename(imgname + '.png', imgname + '.png')
        else:
            os.rename(imgname + '.jpg', imgname + '.png')



