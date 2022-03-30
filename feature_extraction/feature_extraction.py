import os
import os.path as op
from matplotlib.pyplot import box
import numpy as np
import cv2
import torch
import csv
import base64
import json
import urllib.request
from tqdm import tqdm

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

from load_images import get_images_info, get_test_images_info

# data_path = 'data/genome/1600-400-20'
data_path ='data/coco'

coco_classes=list()
with open(op.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        coco_classes.append(object.split(',')[0].lower().strip())

coco_classes_sv=list()
with open(op.join(data_path, 'objects_vocab_sv.txt'), encoding='utf-8') as f:
    for object in f.readlines():
        coco_classes_sv.append(object.split(',')[0].lower().strip())

# MetadataCatalog.get("vg").thing_classes = coco_classes
MetadataCatalog.get("coco").thing_classes = coco_classes

cfg=get_cfg()
# cfg.merge_from_file("../configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml")
cfg.merge_from_file("../configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
# cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
cfg.MODEL.WEIGHTS ="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl"
predictor = DefaultPredictor(cfg)

NUM_OBJECTS = 36

def extract_features(raw_image):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]
        
        # boxes = proposal_boxes[0].tensor

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1): # array([0.5, 0.6, 0.7, 0.8, 0.9])
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:], 
                score_thresh=0.5, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            ) # nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
            if len(ids) == NUM_OBJECTS:
                break
        
        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        return raw_height, raw_width, instances, roi_features.detach().cpu().numpy()

def generate_lineidx_file(filein, idxout):
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos!=fsize:
            tsvout.write(str(fpos)+"\n")
            tsvin.readline()
            fpos = tsvin.tell()
    os.rename(idxout_tmp, idxout)

def write_features_tsv(f: str, image_ids: list, features: list, mode: str):
    assert len(image_ids)==len(features)
    rows=list()
    for i in range(len(image_ids)):
        encoded_features=base64.b64encode(features[i]).decode("utf-8")
        num_boxes=features[i].shape[0]
        row=[image_ids[i], num_boxes, encoded_features]
        rows.append(row)
    
    with open(f, mode=mode) as out_file:
        tsv_writer=csv.writer(out_file, delimiter='\t')
        tsv_writer.writerows(rows)
    
    lineidx = op.splitext(f)[0] + '.lineidx'
    generate_lineidx_file(f, lineidx)

def write_predictions_tsv(f: str, image_ids: list, features: list, instances: list, mode: str):
    assert len(image_ids)==len(features)==len(instances)
    rows=list()
    for i in range(len(image_ids)):
        pred_boxes=instances[i].pred_boxes.tensor.detach().cpu().numpy().tolist()
        classes=list()
        for c in instances[i].pred_classes:
            classes.append(coco_classes_sv[c])
        # print(instances.get_fields())
        # print(instances)
        num_instances=features[i].shape[0]
        image_h=instances[i]._image_size[0]
        image_w=instances[i]._image_size[1]
        confidences=instances[i].scores
        results=dict()
        results["image_h"]=image_h
        results["image_w"]=image_w
        results["num_boxes"]=num_instances
        objects=list()
        for j in range(num_instances):
            d=dict()
            d["class"]=classes[j]
            d["conf"]=confidences.detach().cpu().numpy()[j]
            d["rect"]=pred_boxes[j]
            objects.append(d)
        results["objects"]=objects
        row=[image_ids[i], results]
        rows.append(row)

    with open(f, mode=mode) as out_file:
        tsv_writer=csv.writer(out_file, delimiter='\t')
        tsv_writer.writerows(rows)
    
    lineidx = op.splitext(f)[0] + '.lineidx'
    generate_lineidx_file(f, lineidx)

def write_captions_pt(fin: str, fout: str):
    d=dict()
    with open(fin, 'r', encoding='utf-8') as tsv_file:
        rows=tsv_file.readlines()
    for row in rows:
        row=row.split('\t')
        img_id=int(row[0])
        caption=row[1].strip()
        d[img_id]=caption
    torch.save(d, f=fout)

def write_imageid2idx_json(f: str, image_ids: list, mode: str):
    if mode=='a':
        with open(f, 'r') as fin:
            written_d=json.load(fin)
            l=len(written_d)
    elif mode=='w':
        written_d=dict()
        l=0
    cur_d=dict()
    for i in range(len(image_ids)):
        cur_d[image_ids[i]]=i+l
    written_d.update(cur_d)
    with open(f, 'w') as out_file:
        json.dump(written_d, out_file)

def write_coco_tsv(f: str, image_captions: dict, image_captions_sv: dict):
    rows=list()
    for image_id in image_captions.keys():
        captions=image_captions[image_id]
        for caption in captions:
            cap_id=str(caption['caption_id'])
            cap=image_captions_sv[cap_id]
            row=[image_id, cap]
            rows.append(row)
    with open(f, 'w', newline='') as out_file:
        tsv_writer=csv.writer(out_file, delimiter='\t')
        tsv_writer.writerows(rows)
    
    lineidx = op.splitext(f)[0] + '.lineidx'
    generate_lineidx_file(f, lineidx)

def quote_conversion(path):
    with open(path, 'r') as f:  
        text = f.read() 

    converted_text = text.replace("'", '"') 

    with open(path, 'w') as f: 
        f.write(converted_text)

def concat_feature_and_region(raw_height, raw_width, features, boxes):
    boxes_widths=boxes[:, 2]-boxes[:, 0]
    boxes_heights=boxes[:, 3]-boxes[:, 1]
    scaled_width=boxes_widths/raw_width
    scaled_heights=boxes_heights/raw_height
    scaled_x=boxes[:, 0]/raw_width
    scaled_y=boxes[:, 1]/raw_height
    scaled_width=scaled_width[..., np.newaxis]
    scaled_heights=scaled_heights[..., np.newaxis]
    scaled_x=scaled_x[..., np.newaxis]
    scaled_y=scaled_y[..., np.newaxis]
    spatial_features=np.concatenate((scaled_x, scaled_y, scaled_x+scaled_width, scaled_y+scaled_heights, scaled_width, scaled_heights), axis=1)
    full_features=np.concatenate((features, spatial_features), axis=1)
    return full_features

def get_img_from_img_infos(image_infos: list, directory: str):
    if not op.exists(directory):
        os.mkdir(directory)
        print('Getting images from web...')
        for image_info in tqdm(image_infos):
            url=image_info[0]
            image_id=image_info[3]
            for i in range(5):
                try:
                    urllib.request.urlretrieve(url=url, filename=op.join(directory, str(image_id)+'.jpg'))
                    break
                except:
                    pass
            else:
                raise ValueError("couldn't download image {} after {} times".format(image_id, i+1))

def main():
    # data_dir='data/images'
    data_dir='data/coco_images/train'
    # data_dir='data/coco_images/test'
    TRAIN_NUM=50000
    VAL_NUM=5000
    TEST_NUM=5000

    if not op.exists(data_dir):
        train_img_infos=get_images_info(instance_file='coco/coco-2017/instances_train2017.json', num=TRAIN_NUM)
        # test_img_infos=get_test_images_info(instance_file='coco/coco-2017/instances_train2017.json', n_train=TRAIN_NUM, n_test=TEST_NUM)
        get_img_from_img_infos(image_infos=train_img_infos, directory=data_dir)
    with open('image_captions_train2017.json', 'r') as fp:
        captions = json.load(fp)
    with open('image_captions_train2017_sv.json', 'r', encoding='utf-8') as fp:
        captions_sv=json.load(fp)
    write_coco_tsv(f='my_coco.tsv', image_captions=captions, image_captions_sv=captions_sv)

    images=[op.join(data_dir, f) for f in os.listdir(data_dir) if op.isfile(op.join(data_dir, f))]
    # im = cv2.imread("data/images/input.jpg") # dim 3: h, w, color channel

    image_ids=list()
    all_instances=list()
    all_full_features=list()

    i, all_i=0, 0

    print('Reading images from folder {}'.format(data_dir))
    for image in tqdm(images[all_i:]):
        if i>=500:
            all_i+=i
            if all_i==i:
                mode='w'
            else:
                mode='a'
            
            i=0

            write_features_tsv(f='features.tsv', image_ids=image_ids, features=all_full_features, mode=mode)
            write_predictions_tsv(f='predictions.tsv', image_ids=image_ids, features=all_full_features, instances=all_instances, mode=mode)
            write_imageid2idx_json(f='imageid2idx.json', image_ids=image_ids, mode=mode)
            print('all_i=', all_i)

            image_ids=list()
            all_instances=list()
            all_full_features=list()

        im=cv2.imread(image)
        raw_height, raw_width, instances, features = extract_features(im)
        boxes=instances.pred_boxes.to_numpy()
        full_features=concat_feature_and_region(raw_height, raw_width, features, boxes)
        
        image_ids.append(op.splitext(op.basename(image))[0])
        all_instances.append(instances)
        all_full_features.append(full_features)

        i+=1
    
    # the last 500 items
    write_features_tsv(f='features.tsv', image_ids=image_ids, features=all_full_features, mode=mode)
    write_predictions_tsv(f='predictions.tsv', image_ids=image_ids, features=all_full_features, instances=all_instances, mode=mode)
    write_imageid2idx_json(f='imageid2idx.json', image_ids=image_ids, mode=mode)
    
    # write_captions_pt(fin='extracted_features/train/my_coco.tsv', fout='train_captions.pt')
    quote_conversion('predictions.tsv')    
    print('writing done')

if __name__=='__main__':
    main()