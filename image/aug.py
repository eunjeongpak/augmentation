import os
from os import listdir
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import imgaug as ia
from imgaug import augmenters as iaa
import argparse

def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def read_annotation(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bounding_box_list = []

    file_name = root.find('filename').text
    for obj in root.iter('object'):
        object_label = obj.find("name").text
        for box in obj.findall("bndbox"):
            x_min = int(box.find("xmin").text)
            y_min = int(box.find("ymin").text)
            x_max = int(box.find("xmax").text)
            y_max = int(box.find("ymax").text)

        bounding_box = [object_label, x_min, y_min, x_max, y_max]
        bounding_box_list.append(bounding_box)

    return bounding_box_list, file_name

def read_train_dataset(dir):
    images = []
    annotations = []

    for file in listdir(dir):
        if 'jpg' in file.lower():
            images.append(cv2.imread(dir + file, 1))
            annotation_file = file.replace(file.split('.')[-1], 'xml')
            bounding_box_list, file_name = read_annotation(dir + annotation_file)
            annotations.append((bounding_box_list, annotation_file, file_name))

    images = np.array(images, dtype=object)

    return images, annotations

def aug_code(dir: str,
              new_dir: str,
              method: str):

    ia.seed(1)
    images, annotations = read_train_dataset(dir)

    for idx in range(len(images)):
        image = images[idx]
        boxes = annotations[idx][0]

        ia_bounding_boxes = []
        for box in boxes:
            ia_bounding_boxes.append(ia.BoundingBox(x1=box[1], y1=box[2], x2=box[3], y2=box[4]))
        bbs = ia.BoundingBoxesOnImage(ia_bounding_boxes, shape=image.shape)

        if method == 'rn':
            # rotate & noise
            seq = iaa.Sequential([
                iaa.Rot90(1),
                iaa.AddElementwise((-20, 20), per_channel=0.5)
            ])

        if method == 'fc':
            # flip & color
            seq = iaa.Sequential([
                iaa.Fliplr(1),
                iaa.Flipud(0.2),
                iaa.Multiply((1.2, 1.2))
            ])

        if method == 'co':
            # color
            seq = iaa.Sequential([
                iaa.WithBrightnessChannels(
                    iaa.Add(-40), from_colorspace=iaa.CSPACE_BGR)
            ])

        if method == 'ts':
            # translation & shearing
            seq = iaa.Sequential([
                iaa.Affine(
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    shear=(-16, 16)
                )
            ])

        if method == 'cr':
            # crop & resize & rotate
            seq = iaa.Sequential([
                iaa.Rotate(90),
                iaa.Crop(px=(20, 50), keep_size=True)
            ])

        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        new_image_file = new_dir + method + annotations[idx][2]
        cv2.imwrite(new_image_file, image_aug)

        h, w = np.shape(image_aug)[0:2]

        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = method + annotations[idx][2]
        size_element = ET.SubElement(root, "size")
        ET.SubElement(size_element, "width").text = str(w)
        ET.SubElement(size_element, "height").text = str(h)
        ET.SubElement(size_element, "depth").text = str(3)

        for i in range(len(bbs_aug.bounding_boxes)):
            bb_box = bbs_aug.bounding_boxes[i]
            obj_element = ET.SubElement(root, "object")
            ET.SubElement(obj_element, "name").text = boxes[i][0]
            bndbox_element = ET.SubElement(obj_element, "bndbox")
            ET.SubElement(bndbox_element, "xmin").text = str(int(bb_box.x1))
            ET.SubElement(bndbox_element, "ymin").text = str(int(bb_box.y1))
            ET.SubElement(bndbox_element, "xmax").text = str(int(bb_box.x2))
            ET.SubElement(bndbox_element, "ymax").text = str(int(bb_box.y2))

        xml_str = ET.tostring(root, encoding='utf-8').decode('utf-8')
        new_xml_file = new_dir + method + annotations[idx][1]
        with open(new_xml_file, "w", encoding='utf-8') as f:
            f.write(xml_str)

def main():
    parser = argparse.ArgumentParser(
        description='IMAGE AUGMENTATION')
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--new_dir', type=str, default=None)
    parser.add_argument('--method', type=str, default=None)

    args = parser.parse_args()

    make_dir(args.new_dir)

    aug_code(dir = args.dir,
              new_dir = args.new_dir,
              method = args.method)

if __name__ == "__main__":
    main()


