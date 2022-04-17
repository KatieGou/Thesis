import argparse
import os
import base64
import random
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model.utils.tsv_file import TSVFile
from model.utils.misc import mkdir, set_seed
from model.modeling.modeling_bert import ImageBertForSequenceClassification
from pytorch_transformers import BertTokenizer, BertConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule

class RetrievalDataset(Dataset):
    def __init__(self, tokenizer, args, split: str, is_train: bool) -> None:
        super().__init__()
        self.data_dir=os.path.join(args.data_dir, split)
        self.img_file = os.path.join(self.data_dir, 'features.tsv') # features.tsv: img_id, num_box, features
        caption_file = os.path.join(self.data_dir, '{}_captions.pt'.format(split))
        self.img_tsv = TSVFile(self.img_file)
        self.captions = torch.load(caption_file) # {img_id(int): caption(str)}
        self.img_keys = list(self.captions.keys()) # int
        
        imgid2idx_file = os.path.join(self.data_dir, 'imageid2idx.json') # str: int
        self.image_id2idx = json.load(open(imgid2idx_file))

        if args.add_od_labels:
            label_file = os.path.join(self.data_dir, "predictions.tsv") # img_id, img_info
            self.label_tsv = TSVFile(label_file)
            self.labels=dict() # store img info, {int: info}
            for line_no in range(self.label_tsv.num_rows()):
                row = self.label_tsv.seek(line_no)
                image_id=row[0] # str
                if int(image_id) in self.img_keys:
                    results = json.loads(row[1])
                    objects = results['objects'] if type(results) == dict else results
                    self.labels[int(image_id)]={
                        "image_h": results["image_h"] if type(results) == dict else 600,
                        "image_w": results["image_w"] if type(results) == dict else 800,
                        "class": [cur_d['class'] for cur_d in objects],
                        "boxes": np.array([cur_d['rect'] for cur_d in objects], dtype=np.float32)
                    }
            self.label_tsv._fp.close()
            self.label_tsv._fp = None
        
        if is_train:
            self.num_captions_per_img = args.num_captions_per_img_train
        else:
            self.num_captions_per_img = args.num_captions_per_img_val
            self.has_caption_indexs = False
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_length
        self.max_img_seq_len = args.max_img_seq_length
        self.args = args
    
    def get_od_labels(self, img_key: int):
        if self.args.add_od_labels:
            if type(self.labels[img_key]) == str: # class only
                od_labels = self.labels[img_key]
            else:
                od_labels = ' '.join(self.labels[img_key]['class'])
            return od_labels # str separated by space
    
    def tensorize_example(self, text_a, img_feat, text_b=None, cls_token_segment_id=0, pad_token_segment_id=0, sequence_a_segment_id=0, sequence_b_segment_id=1):
        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_len - 2:
            tokens_a = tokens_a[:self.max_seq_len-2]
        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)

        if seq_a_len<self.max_seq_len and text_b is not None:
            tokens_b=self.tokenizer.tokenize(text_b)
            if len(tokens_b)>self.max_seq_len - seq_a_len - 1:
                tokens_b = tokens_b[: self.max_seq_len - seq_a_len - 1] if self.max_seq_len - seq_a_len - 1>0 else []
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        
        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len # if seq_padding_len<=0, [x]*seq_padding_len=[]
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[:self.max_img_seq_len, :]
            img_len = self.max_img_seq_len
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, self.args.img_feature_dim))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
        
        attention_mask = [1] * seq_len + [0] * seq_padding_len + [1] * img_len + [0] * img_padding_len
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return input_ids, attention_mask, segment_ids, img_feat
    
    def get_image(self, image_id: int):
        image_idx = self.image_id2idx[str(image_id)]
        row = self.img_tsv.seek(image_idx)
        num_boxes = int(row[1])
        try:
            features=np.frombuffer(base64.b64decode(row[-1]), dtype=np.float32).reshape((num_boxes, -1))
        except ValueError:
            features=np.random.rand(num_boxes, self.args.img_feature_dim).astype('float32')
        t_features=torch.from_numpy(features)
        return t_features
    
    def __len__(self):
        return len(self.img_keys) * self.num_captions_per_img
    
    def __getitem__(self, index): # allow its instances to use the [] (indexer) operators.
        if self.is_train:
            img_key = self.img_keys[index] # int
            feature = self.get_image(img_key)
            caption = self.captions[img_key]
            od_labels = self.get_od_labels(img_key)
            example = self.tensorize_example(caption, feature, text_b=od_labels) # example: input_ids, attention_mask, segment_ids, img_feat

            # negative
            neg_img_indexs = list(range(index)) + list(range(index + 1, len(self.img_keys)))
            img_idx_neg = random.choice(neg_img_indexs)
            if random.random() <= 0.5: # caption from a different image.
                caption_neg = self.captions[self.img_keys[img_idx_neg]]
                example_neg = self.tensorize_example(caption_neg, feature, text_b=od_labels)
            else: # randomly select a negative image
                feature_neg = self.get_image(self.img_keys[img_idx_neg])
                od_labels_neg = self.get_od_labels(self.img_keys[img_idx_neg])
                example_neg = self.tensorize_example(caption, feature_neg, text_b=od_labels_neg)
            
            example_pair = tuple(list(example) + [1] + list(example_neg) + [0])
            return index, example_pair
        else:
            img_key = self.img_keys[index]
            feature = self.get_image(img_key)
            caption = self.captions[img_key]
            od_labels = self.get_od_labels(img_key)
            example = self.tensorize_example(caption, feature, text_b=od_labels)
            
            random_num=random.random()
            if random_num <= 0.5: # positive example
                label=1
                return index, tuple(list(example) + [label])
            else:
                label=0
                neg_img_indexs = list(range(index)) + list(range(index + 1, len(self.img_keys)))
                img_idx_neg = random.choice(neg_img_indexs)
                if random_num <= 0.75: # negative caption
                    caption_neg = self.captions[self.img_keys[img_idx_neg]]
                    example_neg = self.tensorize_example(caption_neg, feature, text_b=od_labels)
                    return index, tuple(list(example_neg) + [label])
                else: # negative image
                    feature_neg = self.get_image(self.img_keys[img_idx_neg])
                    od_labels_neg = self.get_od_labels(self.img_keys[img_idx_neg])
                    example_neg = self.tensorize_example(caption, feature_neg, text_b=od_labels_neg)
                    return index, tuple(list(example_neg) + [label])

def compute_score_with_logits(logits, labels):
    if logits.shape[1]>1:
        logits=torch.max(logits, 1)[1].data # indices of max item along dim 1
        scores=logits==labels # tensor of bool
    else:
        scores = torch.zeros_like(labels).cuda() # labels: tensor
        for i, (logit, label) in enumerate(zip(logits, labels)):
            logit_ = torch.sigmoid(logit)
            if (logit_ >= 0.5 and label == 1) or (logit_ < 0.5 and label == 0):
                scores[i] = 1
    return scores

def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(epoch, global_step))
    mkdir(checkpoint_dir)
    if hasattr(model, 'module'):
        model_to_save = model.module
    else:
        model_to_save=model
    model_to_save.save_pretrained(checkpoint_dir) # pytorch_model.bin
    torch.save(args, os.path.join(checkpoint_dir, 'training_args.bin'))
    tokenizer.save_pretrained(checkpoint_dir)
    logger.info("Save checkpoint to {}".format(checkpoint_dir))

def evaluate(args, model, val_dataloader, length: int): # for validation
    model.eval()
    softmax = nn.Softmax(dim=1)
    total_sores=0.0
    for _, batch in tqdm(val_dataloader): # len(indexs)=batch size, seqeuential
        batch = tuple(t.to(args.device) for t in batch)
        labels=batch[4]
        with torch.no_grad(): # disable gradient calculation for inference
            inputs={
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats': batch[3],
                'labels': labels
            } # feed into forward function
            _, logits = model(**inputs)[:2] # loss, logits: (batch_size, num_labels)
            if args.num_labels == 2:
                result = softmax(logits)
            else:
                result = logits
            scores=compute_score_with_logits(result, labels)
            total_sores+=scores.sum().item()
    return total_sores/length # accuracy

def train(args, train_dataset, val_dataset, model, tokenizer): # need to modify to restore training process
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) # args.per_gpu_train_batch_size
    train_sampler=RandomSampler(train_dataset) # Samples elements randomly
    train_dataloader=DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers) # Combines a dataset and a sampler, and provides an iterable over the given dataset

    if args.max_steps>0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs # 50000/args.train_batch_size*30=50000/8*30=187500
    
    if val_dataset is not None:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)
    
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay}, # model.named_parameters(): [name (str), value (tensor)]
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params=grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon) # params (iterable): an iterable of :class:'torch.Tensor's or :class:'dict's. Specifies what Tensors should be optimized
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps) # Linearly increases lr from 0 to 1 over 'warmup_steps' training steps. Keep lr schedule equal to 1 after warmup_steps
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total) # Linearly increases lr from 0 to 1 over 'warmup_steps' training steps. Linearly decreases lr from 1 to 0 over remaining t_total-warmup_steps steps
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc =0,  0.0, 0.0
    model.zero_grad() # clears grad for every parameter
    log_json = []
    for epoch in range(int(args.num_train_epochs)):
        for step, (_, batch) in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids':      torch.cat((batch[0], batch[5]), dim=0),
                'attention_mask': torch.cat((batch[1], batch[6]), dim=0),
                'token_type_ids': torch.cat((batch[2], batch[7]), dim=0), # segment ids
                'img_feats':      torch.cat((batch[3], batch[8]), dim=0),
                'labels':         torch.cat((batch[4], batch[9]), dim=0) # [1*args.train_batch_size, 0*args.train_batch_size]
            }
            outputs = model(**inputs) # (loss, logits, all_hidden_states(optional), all_attentions(optional))
            loss, logits = outputs[:2]
            if args.n_gpu > 1: 
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps # normalize loss
            loss.backward() # computes gradient for every parameter
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm) # if norm exceeds max_norm, the values in the vector will be rescaled so the norm of the vector equals max_norm
            batch_score = compute_score_with_logits(logits, inputs['labels']).sum() # 0 or 1
            batch_acc = batch_score.item() / (args.train_batch_size * 2) # train_batch_size positive and train_batch_size negative
            global_loss += loss.item()
            global_acc += batch_acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                optimizer.step() # apply grad of parameters
                scheduler.step() # update lr
                model.zero_grad() # clears grad for every parameter
                if global_step % args.logging_steps == 0:
                    logger.info(
                        "Epoch: {}, global_step: {}, lr: {:.6f}, batch loss: {:.4f}, global loss: {:.4f}, batch acc: {:.2%}, global acc: {:.2%}".format(
                            epoch, global_step, optimizer.param_groups[0]["lr"], loss, global_loss / global_step, batch_acc, global_acc / global_step
                        )
                    )
                
                if (args.save_steps > 0 and global_step % args.save_steps == 0) or global_step == t_total:
                    save_checkpoint(model, tokenizer, args, epoch, global_step)
                    if args.evaluate_during_training: 
                        logger.info("Perform evaluation at step: %d" % (global_step))
                        logger.info("Num validation examples = %d", len(val_dataset))
                        logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
                        val_acc = evaluate(args, model, val_dataloader, len(val_dataset))
                        epoch_log = {
                            'epoch': epoch, 
                            'global_step': global_step, 
                            'validation acuracy': val_acc
                        }
                        logger.info(
                            'Validation Result: epoch: {}, global_step: {}, validation accuracy: {:.2%}'.format(
                                epoch, global_step, val_acc
                            )
                        )
                        log_json.append(epoch_log)
                        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                            json.dump(log_json, fp) 
    return global_step, global_loss / global_step

def restore_training_settings(args):
    assert not args.do_train and (args.do_test)
    train_args = torch.load(os.path.join(args.eval_model_dir, 'training_args.bin')) # Loads an object saved with torch.save from a file
    override_params = ['do_lower_case', 'img_feature_type', 'max_seq_length', 'max_img_seq_length', 'add_od_labels', 'use_img_layernorm', 'img_layer_norm_eps']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param, test_v, train_v))
                setattr(args, param, train_v)
    return args

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--img_feature_type", default='faster_rcnn', type=str, help="faster_rcnn or mask_rcnn")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False, help="Path to pre-trained model or model type. required for training.")
    parser.add_argument("--output_dir", default='relation_output/', type=str, required=False, help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str, help="Loss function types: support kl, sfmx")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--add_od_labels", default=True, action='store_true', help="Whether to add object detection labels or not.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=36, type=int, help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension, features+location.")
    parser.add_argument("--use_img_layernorm", type=int, default=1, help="Normalize image features with bertlayernorm")
    parser.add_argument("--img_layer_norm_eps", default=1e-12, type=float, help="The epsilon in image feature layer normalization")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_labels", default=2, type=int, help="2 for classification.")
    parser.add_argument("--num_captions_per_img_train", default=1, type=int, help="number of positive matched captions for each training image.")
    parser.add_argument("--num_captions_per_img_val", default=1, type=int, help="number of captions for each validation image.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.") # L2 regularization
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.") # clip the gradients by multiplying the unit vector of the gradients with the threshold
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--num_workers", default=6, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1, help="Save checkpoint every X steps. Will also perform validation.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Run evaluation during training at each save_steps.")
    parser.add_argument("--eval_model_dir", type=str, default='', help="Model directory for evaluation.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization.")
    args=parser.parse_args()

    args.data_dir=os.path.join(*['data', args.img_feature_type, 'coco'])

    assert (args.do_train)^(args.do_test), "do_train and do_test must be set exclusively."

    global logger
    logger=logging.getLogger(__name__)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - [%(name)s - %(filename)s:%(lineno)d] - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    mkdir(args.output_dir)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    # set_seed(args.seed, args.n_gpu)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    config_class, tokenizer_class = BertConfig, BertTokenizer
    model_class = ImageBertForSequenceClassification
    if args.do_train:
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=args.num_labels, finetuning_task='ir')
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.img_layer_norm_eps = args.img_layer_norm_eps
        config.use_img_layernorm = args.use_img_layernorm
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        model.to(args.device)
        logger.info("Training parameters %s", args)
        train_dataset = RetrievalDataset(tokenizer, args, 'train', is_train=True)
        if args.evaluate_during_training:
            val_dataset = RetrievalDataset(tokenizer, args, 'val', is_train=False)
        else:
            val_dataset = None
        global_step, avg_loss = train(args, train_dataset, val_dataset, model, tokenizer)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)
    else:
        checkpoint = args.eval_model_dir
        assert os.path.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint) # config.json
        tokenizer = tokenizer_class.from_pretrained(checkpoint) # special_tokens_map.json and added_tokens.json
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config) # pytorch_model.bin
        model.to(args.device)
        logger.info("Evaluateing parameters %s", args)
        args=restore_training_settings(args)
        test_dataset=RetrievalDataset(tokenizer, args, 'test', is_train=False)
        args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size, num_workers=args.num_workers)
        test_acc = evaluate(args, model, test_dataloader, len(test_dataset))
        logger.info('Testing Accuracy: {:.2%}'.format(test_acc))
        logger.info('Testing done')

if __name__ == "__main__":
    main()