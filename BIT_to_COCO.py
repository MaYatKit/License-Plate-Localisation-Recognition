import json
import codecs
import xml.etree.cElementTree as et
from os import listdir
from shutil import copyfile



class Script:
    bit_vehicle = "/Users/jimmyma/Documents/study/789/openvino_training_extension/openvino_training_extensions/data/bitvehicle/bitvehicle_test_512.json"
    COCO_annotation = "/Users/jimmyma/Documents/study/789/openvino_training_extension/openvino_training_extensions/data/bitvehicle/Annotations_512_test/"
    test_annotation = "/Users/jimmyma/Documents/study/789/openvino_training_extension/openvino_training_extensions/data/bitvehicle/test_annotation"
    test_image = "/Users/jimmyma/Documents/study/789/openvino_training_extension/openvino_training_extensions/data/bitvehicle/test_image/"
    image = "/Users/jimmyma/Documents/study/789/openvino_training_extension/openvino_training_extensions/data/bitvehicle/images/"

    ann_path = "/Users/jimmyma/Documents/study/789/openvino_training_extension/openvino_training_extensions/data/bitvehicle/bitvehicle_test.json"
    ann_path_result = "/Users/jimmyma/Documents/study/789/openvino_training_extension/openvino_training_extensions/data/bitvehicle/bitvehicle_test.json"



def generateCOCOFromJson():
    print("Script started ... ")
    with codecs.open(Script.bit_vehicle,'r', 'utf-8-sig') as j:
        json_test = json.load(j)
        print(json_test)
        for image in json_test['images']:
            root = et.Element('annotation')
            folder = et.SubElement(root, 'folder')
            folder.text = 'bitvehicle'
            filename = et.SubElement(root, 'filename')
            filename.text = image['file_name']
            source = et.SubElement(root, 'source')
            database = et.SubElement(source, 'database')
            database.text = 'BitVehicle'
            annotation = et.SubElement(source, 'annotation')
            annotation.text = 'BitVehicle'
            ann_image = et.SubElement(source, 'image')
            ann_image.text = 'flickr'

            size = et.SubElement(root, 'size')
            width = et.SubElement(size, 'width')
            width.text = str(image['width'])
            height = et.SubElement(size, 'height')
            height.text = str(image['height'])
            depth = et.SubElement(size, 'depth')
            depth.text = '3'

            segmented = et.SubElement(root, 'segmented')
            segmented.text = '0'

            for index in range(len(json_test['annotations'])):
                if json_test['annotations'][index]['image_id'] == image['id']:
                    object = et.SubElement(root, 'object')
                    name = et.SubElement(object, 'name')
                    name.text = 'bg' if json_test['annotations'][index]['category_id'] == 0 else 'plate'
                    pose = et.SubElement(object, 'pose')
                    pose.text = 'Unspecified'
                    truncated = et.SubElement(object, 'truncated')
                    truncated.text = '1' if json_test['annotations'][index]['is_occluded'] else '0'
                    difficult = et.SubElement(object, 'difficult')
                    difficult.text = '0'
                    bndbox = et.SubElement(object, 'bndbox')
                    xmin = et.SubElement(bndbox, 'xmin')
                    xmin.text = str(json_test['annotations'][index]['bbox'][0])
                    ymin = et.SubElement(bndbox, 'ymin')
                    ymin.text = str(json_test['annotations'][index]['bbox'][1])
                    xmax = et.SubElement(bndbox, 'xmax')
                    xmax.text = str(json_test['annotations'][index]['bbox'][0] + json_test['annotations'][index]['bbox'][2])
                    ymax = et.SubElement(bndbox, 'ymax')
                    ymax.text = str(json_test['annotations'][index]['bbox'][1] + json_test['annotations'][index]['bbox'][3])
                    if index < (len(json_test['annotations']) - 1) and json_test['annotations'][index + 1]['image_id'] != image['id']:
                        break
            tree = et.ElementTree(root)
            tree.write(Script.COCO_annotation + image['file_name'].split('.')[0] + '.xml')



def moveImage():
    for file in listdir(Script.test_annotation):
        image_name = file.split('.')[0]
        copyfile(Script.image + image_name + '.jpg', Script.test_image + image_name + '.jpg')




if __name__ == '__main__':
    generateCOCOFromJson()
