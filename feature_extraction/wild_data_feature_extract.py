import glob
import os
import os.path as op
import PIL.Image
import json
import torch
import cv2

from feature_extract import extract_features, concat_feature_and_region, write_features_tsv, write_predictions_tsv, quote_conversion

def convert_png_jpeg_to_jpg(file: str):
    if file.endswith('.png'):
        jpg_file=file.replace('.png', '.jpg')
    else:
        jpg_file=file.replace('.jpeg', '.jpg')
    img=PIL.Image.open(file).convert('RGB')
    img.save(jpg_file)
    os.remove(file)

def write_imageid2idx_json(data_dir: str):
    imageid2idx=dict()
    images=[op.join(data_dir, f) for f in os.listdir(data_dir) if op.isfile(op.join(data_dir, f)) and f.endswith('.jpg')]
    image_ids=[op.splitext(op.basename(image))[0] for image in images]
    for i in range(len(image_ids)):
        imageid2idx[image_ids[i]]=i
    with open('imageid2idx.json', 'w') as f:
        json.dump(imageid2idx, f)
    return image_ids

def write_captions_pt(data_dir: str, image_ids: list):
    d=dict()
    for image_id in image_ids:
        img_id=int(image_id)
        with open(op.join(data_dir, image_id+'.txt'), 'r', encoding='utf-8') as f:
            caption=f.read()
        d[img_id]=caption
    torch.save(d, 'test_captions.pt')

def main():
    data_dir='data/wild_data'
    
    '''Write imageid2idx.json and test_captions.pt'''
    for file in glob.glob(op.join(data_dir, '*.png'))+glob.glob(op.join(data_dir, '*.jpeg')):
        convert_png_jpeg_to_jpg(file)
    image_ids=write_imageid2idx_json(data_dir)
    write_captions_pt(data_dir, image_ids)

    all_instances=list()
    all_full_features=list()
    images=[op.join(data_dir, f) for f in os.listdir(data_dir) if op.isfile(op.join(data_dir, f)) and f.endswith('.jpg')]
    for image in images:
        raw_image=cv2.imread(image)
        raw_height, raw_width, instances, features=extract_features(raw_image)
        boxes=instances.pred_boxes.to_numpy()
        full_features=concat_feature_and_region(raw_height, raw_width, features, boxes)
        all_instances.append(instances)
        all_full_features.append(full_features)
    write_features_tsv(f='features.tsv', image_ids=image_ids, features=all_full_features, mode='w')
    write_predictions_tsv(f='predictions.tsv', image_ids=image_ids, features=all_full_features, instances=all_instances, mode='w')
    quote_conversion('predictions.tsv')    
    print('writing done')

if __name__=='__main__':
    main()