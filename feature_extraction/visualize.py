import os.path as op
import numpy as np
import cv2
import torch
import os, random

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

from feature_extract import extract_features
data_path ='data/coco'
coco_classes=list()
with open(op.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        coco_classes.append(object.split(',')[0].lower().strip())

coco_classes_sv=list()
with open(op.join(data_path, 'objects_vocab_sv.txt'), encoding='utf-8') as f:
    for object in f.readlines():
        coco_classes_sv.append(object.split(',')[0].lower().strip())

MetadataCatalog.get("coco").thing_classes = coco_classes
cfg=get_cfg()
# cfg.merge_from_file("../configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml")
cfg.merge_from_file("../configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml") # faster rcnn
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
# cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
cfg.MODEL.WEIGHTS ="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl" # faster rcnn
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

def save_objects(raw_image, instances, filepath):
    pred=instances.to('cpu')
    v=Visualizer(raw_image[:, :, :], MetadataCatalog.get("coco"), scale=1.2)
    v = v.draw_instance_predictions(pred)
    v.save(filepath)
    print('saved to', filepath)

def main():
    if not os.path.exists('detected_objects'):
        os.makedirs('detected_objects')
    for i in range(10):
        img=os.path.join('data/coco_images/train', random.choice(os.listdir('data/coco_images/train')))
        raw_image=cv2.imread(img)
        _, _, instances, _ = extract_features(raw_image)
        save_objects(raw_image, instances, os.path.join(*['detected_objects', 'img_result_'+str(i)+'.jpg']))

if __name__ == "__main__":
    main()