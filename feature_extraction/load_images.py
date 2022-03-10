from pycocotools.coco import COCO
import json
from tqdm import tqdm
import os

def get_images_info(instance_file: str, num: int) -> list:
    with open(instance_file) as json_file:
        d=json.load(json_file)
    images=d['images']
    info=list()
    i=0
    for image in images:
        if i>=num:
            break
        l=[image['coco_url'], image['height'], image['width'], image['id']]
        info.append(l)
        i+=1
    return info # [[url, height, width, img_id]...]

def get_image_captions(caption_file: str, image_infos: list, num: int):
    coco_caps=COCO(caption_file)
    d_images=dict()
    i=0
    print('Loading image captions...')
    for image_info in tqdm(image_infos):
        if i>=num:
            break
        img_id=image_info[3]
        d_images[img_id]=list()
        ann_ids=coco_caps.getAnnIds(img_id)
        anns=coco_caps.loadAnns(ann_ids)
        ann=anns[0]
        ann.pop('image_id')
        ann['caption_id']=ann['id']
        ann.pop('id')
        d_images[img_id].append(ann)
        i+=1
    f='image_'+os.path.basename(caption_file)
    print('Writing to {}...'.format(f))
    with open(f, 'w') as fp:
        json.dump(d_images, fp)

def get_test_images_info(instance_file: str, n_train: int, n_test: int) -> list:
    with open(instance_file) as json_file:
        d=json.load(json_file)
    images=d['images']
    info=list()
    i=0
    for image in images[n_train:]:
        if i>=n_test:
            break
        l=[image['coco_url'], image['height'], image['width'], image['id']]
        info.append(l)
        i+=1
    return info # [[url, height, width, img_id]...]

def get_test_image_captions(caption_file: str, image_infos: list, num: int):
    coco_caps=COCO(caption_file)
    d_images=dict()
    i=0
    print('Loading image captions...')
    for image_info in tqdm(image_infos):
        if i>=num:
            break
        img_id=image_info[3]
        d_images[img_id]=list()
        ann_ids=coco_caps.getAnnIds(img_id)
        anns=coco_caps.loadAnns(ann_ids)
        ann=anns[0]
        ann.pop('image_id')
        ann['caption_id']=ann['id']
        ann.pop('id')
        d_images[img_id].append(ann)
        i+=1
    f='image_captions_test2017.json'
    print('Writing to {}...'.format(f))
    with open(f, 'w') as fp:
        json.dump(d_images, fp)

def main():
    # info=get_images_info('coco/coco-2017/instances_train2017.json', 50000)

    # get_image_captions('coco/coco-2017/captions_train2017.json', info, 50000)

    info=get_test_images_info('coco/coco-2017/instances_train2017.json', 50000, 5000)
    get_test_image_captions('coco/coco-2017/captions_train2017.json', info, 5000)

if __name__=='__main__':
    main()