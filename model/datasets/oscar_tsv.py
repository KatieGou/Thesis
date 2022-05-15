import os
import time
import json
import logging
import random
import base64
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from model.utils.tsv_file import TSVFile
from model.utils.misc import load_from_yaml_file

class OscarTSVDataset(Dataset):
    """Dataset for pretraining
    """    
    def __init__(self, yaml_file, args=None, tokenizer=None, seq_len=35, encoding="utf-8", corpus_lines=None, **kwargs):
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = os.path.dirname(yaml_file)
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len # max sequence length
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_tsvfile = TSVFile(os.path.join(*[self.root, args.img_feature_type, self.cfg['corpus_file']])) # my_coco.tsv: [img_id, caption]
        if 'textb_sample_mode' in kwargs:
            self.textb_sample_mode = kwargs['textb_sample_mode']
        else:
            self.textb_sample_mode = args.textb_sample_mode
        
        self.datasets_names = self.cfg['corpus'].split('_') # coco only
        logging.info('Datasets: {}'.format(','.join(self.datasets_names)))
        self.image_label_path = self.cfg['image_label_path']
        for key, val in self.image_label_path.items():
            self.image_label_path[key] = os.path.join(*[self.root, args.img_feature_type, val]) # coco
        self.image_feature_path = self.cfg['image_feature_path']
        self.image_file_name = 'features.tsv'
        if args.data_dir is not None:
            for key, val in self.image_feature_path.items():
                if key in self.datasets_names:
                    self.image_feature_path[key] = os.path.join(*[args.data_dir, args.img_feature_type, val])
                else:
                    logging.info("Data {} with path {} is not used in the training.".format(key, val))
        self.encoding = encoding
        self.current_doc = 0
        self.current_img = ''
        self.args = args

        self.sample_counter = 0

        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line, [{doc_id: , line: }, ...]
        
        # load image tags and features
        t_start = time.time()
        self.img_label_file = None
        self.img_label_offset_map = None
        self.img_feature_file = None
        self.img_feat_offset_map = None

        self.load_img_labels() # predictions.tsv and image2idx.json
        self.load_img_tsv_features() # features.tsv
        t_end = time.time()
        logging.info('Info: loading img features using {} secs'.format(t_end - t_start))

        self.all_docs = list() # [[img_id, texta, textb], ...]
        self.imgid2labels = dict()
        self.corpus_lines = 0
        max_tokens = 0
        for line_no in tqdm(range(len(self.corpus_tsvfile))):
            doc=list() # img_id, caption, detected classes
            row = self.corpus_tsvfile.seek(line_no)
            img_id=row[0] # should be str
            dataset_name='coco'
            img_feat_offset_map = self.img_feat_offset_map[dataset_name]
            assert img_id in img_feat_offset_map, 'Image id {} cannot be found in image feature imageid2index file!'.format(img_id)

            # append id info
            doc.append(img_id)
            # append text_a info, caption
            self.corpus_lines+=1
            sample = {"doc_id": len(self.all_docs), "line": len(doc)}
            self.sample_to_doc.append(sample)
            assert len(row[1]) != 0, "Text_a is empty!"
            doc.append(row[1])
            # append text_b info, labels
            self.corpus_lines+=1
            label_id=img_id
            label_line_no = self.img_label_offset_map[dataset_name][label_id]
            rowb = self.img_label_file[dataset_name].seek(label_line_no) # should be [img_id, labels (pred_class, bbox, conf)]
            assert label_id == rowb[0]
            results = json.loads(rowb[1]) # dict, {image_h: , image_w: , num_boxes: , objects: [{class: , conf: , rect: , }, ...]}
            objects = results['objects']
            if row[0] not in self.imgid2labels:
                self.imgid2labels[row[0]]={
                    "image_h": results["image_h"],
                    "image_w": results["image_w"],
                    "boxes": None
                }
            else:
                assert results["image_h"] == self.imgid2labels[row[0]]["image_h"], "Image_h does not match in image {}!".format(row[0])
                assert results["image_w"] == self.imgid2labels[row[0]]["image_w"], "Image_w does not match in image {}!".format(row[0])
            if args.use_gtlabels and 'gt_objects' in results:
                # use ground-truth tags for text_b
                textb = ' '.join([cur_d['class'] for cur_d in results["gt_objects"]])
            else:
                textb = ' '.join([cur_d['class'] for cur_d in objects])
            # assert len(textb) != 0, "Text_b is empty in {} : {}".format(dataset_name, row[1])
            if len(textb)==0:
                textb='ingenting'
            doc.append(textb)

            # add to all_docs
            max_tokens = max(max_tokens, len(doc[1].split(' ')) + len(doc[2].split(' ')))
            self.all_docs.append(doc)
        
        self.num_docs = len(self.all_docs)
        logging.info("Max_tokens: {}".format(max_tokens))
        logging.info("Total docs - Corpus_lines: {}-{}".format(self.num_docs, self.corpus_lines))
    
    def __len__(self):
        return self.corpus_lines - self.num_docs
    
    def get_img_info(self, idx):
        """get image height and width

        Args:
            idx (int): idx you want to check

        Returns:
            dict: image height and width
        """        
        sample = self.sample_to_doc[idx]
        img_id = self.all_docs[sample["doc_id"]][0].strip()
        imgid2labels = self.imgid2labels[img_id]
        return {"height": imgid2labels["image_h"], "width": imgid2labels["image_w"]}
    
    def __getitem__(self, item): # allows its instances to use the [] (indexer) operators
        cur_id = self.sample_counter
        self.sample_counter += 1        
        img_id, t1, t2, is_next_label, is_img_match = self.random_sent(item)

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        if self.args.use_b:
            tokens_b = self.tokenizer.tokenize(t2)
        else:
            tokens_b = None
        
        # combine to one sample
        cur_example=InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label, img_id=img_id, is_img_match=is_img_match)

        # get image feature
        img_feat=self.get_img_feature(image_id=img_id)
        if img_feat.shape[0]>=self.args.max_img_seq_length:
            img_feat = img_feat[:self.args.max_img_seq_length, ]
            img_feat_len = img_feat.shape[0]
        else:
            img_feat_len = img_feat.shape[0]
            padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
        
        # transform sample to features
        cur_features=convert_example_to_features(self.args, cur_example, self.seq_len, self.tokenizer, img_feat_len)

        return img_feat, (
            torch.tensor(cur_features.input_ids, dtype=torch.long),
            torch.tensor(cur_features.input_mask, dtype=torch.long),
            torch.tensor(cur_features.segment_ids, dtype=torch.long),
            torch.tensor(cur_features.lm_label_ids, dtype=torch.long),
            torch.tensor(cur_features.is_next),
            torch.tensor(cur_features.is_img_match)
        ), item
    
    def random_sent(self, index):
        """Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences from one doc. With 50% the second sentence will be a random one from another doc.

        Args:
            index (int): index of sample

        Returns:
            tuple: img_id, sentence1, sentence2, isNextSentence Label (0: match, 1: unmatch), img_match_label (0: match, 1: unmatch)
        """        
        img_id, t1, t2=self.get_corpus_line(index)
        rand_dice=random.random()
        if rand_dice>0.5: # right text pair
            label = 0
            random_img_id = img_id
        elif rand_dice > self.args.texta_false_prob and t2!="": # sample wrong texta or textb
            random_img_id, t2 = self.get_random_line() # random_img_id and t2 corresponds
            label=1
        else: # wrong retrieval triplets
            random_img_id, t1 = self.get_random_texta() # random_img_id and t1 corresponds
            label = self.args.num_contrast_classes-1
        
        img_match_label=0
        if img_id != random_img_id:
            img_match_label = 1
        
        assert len(t1) > 0
        assert len(t2) > 0 or not self.args.use_b
        return img_id, t1, t2, label, img_match_label # (img_id, t1) and (img_id, t2) have at least one corresponds
    
    def get_corpus_line(self, item):
        """Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.

        Args:
            item (int): index of sample.
        
        Returns:
            tuple: img_id, sentence1, sentence2
        """        
        assert item < self.corpus_lines
        sample = self.sample_to_doc[item]
        img_id = self.all_docs[sample["doc_id"]][0].strip()
        t1=self.all_docs[sample["doc_id"]][sample["line"]] # texta
        t2 = self.all_docs[sample["doc_id"]][sample["line"] + 1] # textb
        self.current_doc = sample["doc_id"]
        self.current_img = img_id

        assert t1 != ""
        if self.args.use_b:
            assert t2 != ""
        else:
            t2=""
        return img_id, t1, t2
    
    def get_random_line(self):
        """Get random line from another document for nextSentence task

        Returns:
            tuple: img_id, random sampled text
        """        
        if self.textb_sample_mode in [0, 1]: # sample from all docs
            for _ in range(10):
                rand_doc_idx = random.randrange(0, len(self.all_docs))
                img_id = self.all_docs[rand_doc_idx][0]
                if img_id != self.current_img:
                    break
            rand_doc=self.all_docs[rand_doc_idx]
        img_id = rand_doc[0]
        if self.textb_sample_mode == 0: # default sample mode
            line = rand_doc[random.randrange(1, len(rand_doc))] # texta or textb
        else: # only sample text_b
            line = rand_doc[2]
        return img_id, line
    
    def get_random_texta(self):
        """Get random text_a from another document for nextSentence task.

        Returns:
            tuple: img_id, random sampled texta
        """        
        for _ in range(10):
            rand_doc_idx = random.randrange(0, len(self.all_docs))
            img_id = self.all_docs[rand_doc_idx][0]
            if img_id != self.current_img:
                break
        rand_doc = self.all_docs[rand_doc_idx]
        img_id = rand_doc[0]
        line = rand_doc[1] # texta
        return img_id, line

    def load_img_labels(self):
        self.check_img_label_file()
        self.check_img_label_offset_map()
    
    def check_img_label_file(self):
        if self.img_label_file is None:
            self.img_label_file=dict()
            for dataset_name in self.datasets_names:
                img_label_file_path = os.path.join(self.image_label_path[dataset_name], 'predictions.tsv')
                t_s = time.time()
                self.img_label_file[dataset_name] = TSVFile(img_label_file_path)
                t_e = time.time()
                logging.info("Open image label file {}, time: {}".format(img_label_file_path, (t_e - t_s)))
    
    def check_img_label_offset_map(self):
        if self.img_label_offset_map is None:
            self.img_label_offset_map=dict()
            for dataset_name in self.datasets_names:
                img_label_offset_map_path = os.path.join(self.image_label_path[dataset_name], 'imageid2idx.json')
                t_s = time.time()
                self.img_label_offset_map[dataset_name] = json.load(open(img_label_offset_map_path))
                t_e = time.time()
                logging.info("Load image label offset map: {}, time: {}".format(img_label_offset_map_path, (t_e - t_s)))
    
    def load_img_tsv_features(self):
        if self.img_feature_file is None:
            self.img_feature_file=dict()
            self.img_feat_offset_map=dict()
            for dataset_name in self.datasets_names:
                logging.info("* Loading dataset {}".format(dataset_name))
                t_s = time.time()
                chunk_fp = os.path.join(self.image_feature_path[dataset_name], self.image_file_name)
                self.img_feature_file[dataset_name] = TSVFile(chunk_fp)
                chunk_offsetmap = os.path.join(os.path.dirname(chunk_fp), 'imageid2idx.json')
                assert os.path.isfile(chunk_offsetmap), "Imageid2idx file {} does not exists!".format(chunk_offsetmap)
                self.img_feat_offset_map[dataset_name] = json.load(open(chunk_offsetmap, 'r'))
                t_e = time.time()
                logging.info("Open dataset {}, time: {}".format(chunk_fp, (t_e - t_s)))
    
    def get_img_feature(self, image_id):
        """decode the image feature

        Args:
            image_id (int): image id

        Returns:
            torch.tensor/None: feature of the image
        """        
        self.load_img_tsv_features()
        img_id=image_id
        dataset_name='coco'
        img_feat_offset_map=self.img_feat_offset_map[dataset_name]
        img_feature_file=self.img_feature_file[dataset_name]

        if img_id in img_feat_offset_map:
            img_offset = img_feat_offset_map[img_id]
            arr = img_feature_file.seek(img_offset)
            num_boxes = int(arr[1])
            feat=np.frombuffer(base64.b64decode(arr[-1]), dtype=np.float32).reshape((num_boxes, self.args.img_feature_dim))
            feat=torch.from_numpy(feat)
            return feat
        return None
    
class InputExample:
    """A single training/test example for the language model."""
    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None, img_id=None, is_img_match=None, img_label=None) -> None:
        """Constructs a InputExample.

        Args:
            guid (int): Unique id for the example.
            tokens_a (list): list of tokens of texta
            tokens_b (list, optional): list of tokens of textb. Defaults to None.
            is_next (int, optional): isNextSentence Label. Defaults to None.
            lm_labels (list, optional): masked tokens labels. Defaults to None.
            img_id (int, optional): image id. Defaults to None.
            is_img_match (int, optional): img_match_label. Defaults to None.
            img_label (_type_, optional): _description_. Defaults to None.
        """        
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.img_id = img_id
        self.is_img_match = is_img_match
        self.img_label = img_label

class InputFeatures:
    """A single set of features of data
    """    
    def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids, img_feat_len, is_img_match) -> None:
        self.input_ids=input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.img_feat_len = img_feat_len
        self.is_img_match = is_img_match

def convert_example_to_features(args, example, max_seq_length, tokenizer, img_feat_len):
    """convert an example to information to feed the model
    """    
    tokens_a = example.tokens_a
    tokens_b = None
    if example.tokens_b:
        tokens_b=example.tokens_b
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3) # Account for [CLS], [SEP], [SEP] with "- 3"
    else:
        if len(tokens_a)>max_seq_length-2:
            tokens_a=tokens_a[:(max_seq_length - 2)]
    
    is_next_type = example.is_next * example.is_img_match # is_img_match = 1 for mismatch images
    if args.num_contrast_classes == 2 and args.texta_false_prob == 0.5 and is_next_type==1:
        is_next_type = 2 # is_next_type 0: correct pair, 1: wrong text_b, 2: wrong text_a
    tokens_a, t1_label=random_word(tokens_a, tokenizer)
    if tokens_b:
        if not args.mask_loss_for_unmatched and is_next_type==1:
            t2_label=[-1]*len(tokens_b)
        else:
            tokens_b, t2_label=random_word(tokens_b, tokenizer)
    
    # concatenate lm labels and account for CLS, SEP, SEP
    if tokens_b:
        lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])
    else:
        lm_label_ids = ([-1] + t1_label + [-1])
    
    tokens=list()
    segment_ids=list()
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        assert len(tokens_b)>0
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length
    while len(input_ids)<max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)
    
    assert len(input_ids)==max_seq_length
    assert len(input_mask)==max_seq_length
    assert len(segment_ids)==max_seq_length
    assert len(lm_label_ids)==max_seq_length

    # image features
    if args.max_img_seq_length>0:
        if img_feat_len > args.max_img_seq_length: # img_feat_len: num_boxes
            input_mask = input_mask + [1] * img_feat_len
        else:
            input_mask = input_mask + [1] * img_feat_len
            pad_img_feat_len = args.max_img_seq_length - img_feat_len
            input_mask = input_mask + ([0] * pad_img_feat_len) # input_mask: concate token mask and img mask
    
    lm_label_ids = lm_label_ids + [-1] * args.max_img_seq_length

    if example.guid < 1: # the information is printed num_workers times
        logging.info("*** Example ***")
        logging.info("guid: %s" % example.guid)
        logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("LM label: %s " % lm_label_ids) # index of corresponding word
        logging.info("Is next sentence label: %s " % example.is_next)
    
    features=InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, lm_label_ids=lm_label_ids, is_next=example.is_next, img_feat_len=img_feat_len, is_img_match=example.is_img_match)

    return features

def random_word(tokens, tokenizer):
    """Masking some random tokens for Language Model task with probabilities as in the original BERT paper.

    Args:
        tokens (list): tokenized sentence.
        tokenizer (Tokenizer): object used for tokenization, need it's vocab

    Returns:
        tuple: masked tokens, related labels for LM prediction
    """    
    output_label=list()

    for i, token in enumerate(tokens):
        prob=random.random()
        if prob < 0.15: # mask token with 15% probability
            prob/=0.15
            if prob<0.8: # 80% randomly change token to mask token
                tokens[i]="[MASK]"
            elif prob < 0.9: # 10% randomly change token to random token
                tokens[i]=random.choice(list(tokenizer.vocab.items()))[0]
            # rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError: # For unknown words
                output_label.append(tokenizer.vocab["[UNK]"])
                logging.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else: # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    return tokens, output_label

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length=len(tokens_a)+len(tokens_b)
        if total_length<=max_length:
            break
        if len(tokens_a)>len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()