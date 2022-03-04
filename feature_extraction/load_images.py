# training: 50000 from Train split
# validation: 5000 from Validation split
# test: 5000 from Train split

from pycocotools.coco import COCO
import numpy as np
from skimage import io
import json

TRAIN_NUM=50000
VAL_NUM=5000
TEST_NUM=5000

def get_images_info(instance_file: str) -> list[list[str]]:
    with open(instance_file) as json_file:
        d=json.load(json_file)
    images=d['images']
    info=list()
    for image in images:
        l=[image['coco_url'], image['height'], image['width'], image['id']]
        info.append(l)
    return info # [[url, height, width, img_id]...]

def get_image_captions(caption_file: str, img_ids: list) -> dict:
    coco_caps=COCO(caption_file)
    d_images=dict()
    for img_id in img_ids:
        d_images[img_id]=list()
        ann_ids=coco_caps.getAnnIds(img_id)
        anns=coco_caps.loadAnns(ann_ids)
        for ann in anns:
            ann.pop('image_id')
            ann['caption_id']=ann['id']
            ann.pop('id')
            d_images[img_id].append(ann)
    return d_images

def main():
    info=get_images_info('coco/coco-2017/instances_val2017.json')
    print(len(info))
    print(info[:10])

    d_images=get_image_captions('coco/coco-2017/captions_val2017.json', [info[0][3], info[1][3]])
    print(d_images)

if __name__=='__main__':
    main()