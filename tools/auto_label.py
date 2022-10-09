import os
import cv2

headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>VOC</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>%d</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
tailstr = '''\
</annotation>
'''


def save_annotations(boxes, img, filename):
    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]
    # H,W,C = img.shape
    img_name = filename.split('.')[0] + '.bmp'
    head = headstr % (img_name, W, H, C)  # 写入头文件
    tail = tailstr  # 写入尾文件
    # 写入boxes
    save_path = anno_path + filename.split('.')[0] + '.xml'
    f = open(save_path, 'w')
    f.write(head)
    for box in boxes:
        f.write(objstr % (str(box[0]), 0, float(box[2]), float(box[3]), float(box[2]) + float(box[4]), float(box[3]) + float(box[5])))
    f.write(tail)


if __name__ == '__main__':
    # 设置路径
    root_path = './'
    total_label_path = root_path + 'txt/'  # txt存储的路径
    total_img_path = root_path + 'img/'  # 图像存储路径
    anno_path = root_path + 'Annotations/'  # 存储生成的xml标注文件
    # 判断当前路径下是否存在Annotations这个文件夹,若不存在，自动创建一个
    if not os.path.exists(anno_path):
        os.mkdir(anno_path)
    # 逐个读取txt标注文件
    for filename in os.listdir(total_label_path):
        cur_label_path = total_label_path + filename
        cur_img_path = total_img_path + filename.split('.')[0] + '.jpg'  # 换一下文件名后缀
        cur_boxes = []
        # 读取当前txt文件中的内容
        with open(cur_label_path, 'r') as file:
            while True:
                line = file.readline().strip()  # .strip()用来去掉'\r,\n'
                if not line:
                    break
                line_list = [ele for ele in line.split(' ')]
                cur_boxes.append(line_list)
        # 读取当前图像
        cur_img = cv2.imread(cur_img_path)
        # 进行xml文档存储
        save_annotations(cur_boxes, cur_img, filename)
