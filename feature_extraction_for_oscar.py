import os
import os.path as op
from matplotlib.pyplot import box
import numpy as np
import cv2
import torch
import csv
import base64
import json

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

data_path = 'data/genome/1600-400-20'

vg_classes=list()
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

MetadataCatalog.get("vg").thing_classes = vg_classes

cfg=get_cfg()
cfg.merge_from_file("../configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml")
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
predictor = DefaultPredictor(cfg)

im = cv2.imread("data/images/input.jpg") # dim 3: h, w, color channel
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

def write_features_tsv(f: str, image_id: str, features: np.array):
    encoded_features=base64.b64encode(features).decode("utf-8")
    num_boxes=features.shape[0]
    row=[image_id, num_boxes, encoded_features]
    with open(f, 'w') as out_file:
        tsv_writer=csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(row)
    
    lineidx = op.splitext(f)[0] + '.lineidx'
    generate_lineidx_file(f, lineidx)

def write_predictions_tsv(f: str, image_id: str, features: np.array, instances):
    pred_boxes=instances.pred_boxes.tensor.detach().cpu().numpy().tolist()
    classes=list()
    for c in instances.pred_classes:
        classes.append(vg_classes[c])
    # print(instances.get_fields())
    # print(instances)
    num_instances=features.shape[0]
    image_h=instances._image_size[0]
    image_w=instances._image_size[1]
    confidences=instances.scores
    results=dict()
    results["image_h"]=image_h
    results["image_w"]=image_w
    results["num_boxes"]=num_instances
    objects=list()
    for i in range(num_instances):
        d=dict()
        d["class"]=classes[i]
        d["conf"]=confidences.detach().cpu().numpy()[i]
        d["rect"]=pred_boxes[i]
        objects.append(d)
    results["objects"]=objects
    row=[image_id, results]

    with open(f, 'w') as out_file:
        tsv_writer=csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(row)
    
    lineidx = op.splitext(f)[0] + '.lineidx'
    generate_lineidx_file(f, lineidx)

def write_captions_pt(f: str, image_id: str, captions: list):
    d=dict()
    d[image_id]=captions
    torch.save(d, f=f)

def write_imageid2idx_json(f: str, image_ids: list):
    d=dict()
    for i in range(len(image_ids)):
        d[image_ids[i]]=i
    with open(f, 'w') as out_file:
        json.dump(d, out_file)

def write_coco_tsv(f: str, img_info: str, label_info: str, caption: str):
    row=[img_info, label_info, caption]
    with open(f, 'w') as out_file:
        tsv_writer=csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(row)
    
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

def main():
    raw_height, raw_width, instances, features = extract_features(im)
    boxes=instances.pred_boxes.to_numpy()
    full_features=concat_feature_and_region(raw_height, raw_width, features, boxes)
    # print('full_features.shape:', full_features.shape)
    write_features_tsv(f='features.tsv', image_id='000542', features=full_features)
    write_predictions_tsv(f='predictions.tsv', image_id='000542', features=full_features, instances=instances)
    quote_conversion('predictions.tsv')
    # write_captions_pt(f='train_captions.pt', image_id='000542', captions=['a man in red is riding a horse'])
    write_imageid2idx_json(f='imageid2idx.json', image_ids=["000542"])
    write_coco_tsv(f='my_coco.tsv', img_info='coco_cap_000542', label_info='coco_cap_000542', caption='a man in red is riding a horse')
    print('writing done')

if __name__=='__main__':
    main()
