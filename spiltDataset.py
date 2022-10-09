
import os
import random
import shutil
import time

def copyFile(fileDir, class_name):
    image_list = os.listdir(fileDir) # 获取图片的原始路径
    image_number = len(image_list)

    train_number = int(image_number * train_rate)
    train_sample = random.sample(image_list, train_number) # 从image_list中随机获取0.8比例的图像.
    test_sample = list(set(image_list) - set(train_sample))
    sample = [train_sample, test_sample]

    # 复制图像到目标文件夹
    for k in range(len(save_dir)):
        if os.path.isdir(save_dir[k]):
            for name in sample[k]:
                shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k], name))
        else:
            os.makedirs(save_dir[k] )
            for name in sample[k]:
                shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k], name))

if __name__ == '__main__':
    # ######################################################################
    time_start = time.time()

    # 原始数据集路径
    origion_path = 'D:\\code_cxz\\mixformer\\PaddleDetection\\dataset\\jzxcoco'

    # 保存路径
    save_train_dir = 'D:\\code_cxz\\mixformer\\PaddleDetection\\dataset\\jzxcoco\\train\\'
    save_test_dir = 'D:\\code_cxz\\mixformer\\PaddleDetection\\dataset\\jzxcoco\\val\\'
    save_dir = [save_train_dir, save_test_dir]

    # 训练集比例
    train_rate = 0.8

    # 数据集类别及数量
    file_list = os.listdir(origion_path)
    num_classes = len(file_list)

    for i in range(num_classes):
        class_name = file_list[i]
        image_Dir = os.path.join(origion_path, class_name)
        copyFile(image_Dir, class_name)
        print('%s划分完毕！' % class_name)

    time_end = time.time()
    print('---------------')
    print('训练集和测试集划分共耗时%s!' % (time_end - time_start))


# # ###############################################################################
#     time_start = time.time()
#
#     # 原始数据集路径
#     origion_path = 'D:\\code_cxz\\mixformer\\PaddleClas-c\\dataset\\VOC2007_3.05\\liang'
#
#     # 保存路径
#     save_train_dir = 'D:\\code_cxz\\mixformer\\PaddleClas-c\\dataset\\VOC2007_3.05\\train\\liang\\'
#     save_test_dir = 'D:\\code_cxz\\mixformer\\PaddleClas-c\\dataset\\VOC2007_3.05\\val\\liang\\'
#     save_dir = [save_train_dir, save_test_dir]
#
#     # 训练集比例
#     train_rate = 0.2
#
#     # 数据集类别及数量
#     file_list = os.listdir(origion_path)
#     num_classes = len(file_list)
#
#     for i in range(num_classes):
#         class_name = file_list[i]
#         image_Dir = os.path.join(origion_path, class_name)
#         copyFile(image_Dir, class_name)
#         print('%s划分完毕！' % class_name)
#
#     time_end = time.time()
#     print('---------------')
#     print('良性训练集和测试集划分共耗时%s!' % (time_end - time_start))

