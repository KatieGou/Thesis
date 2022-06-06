import json
import logging
import os
import base64
import argparse
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
import warnings
warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter
from model.utils.misc import mkdir
from model.utils.tsv_file import TSVFile
from lstm_training import SimpleLSTM

from pytorch_transformers import BertTokenizer, BertConfig, AdamW, WarmupLinearSchedule, WarmupConstantSchedule

class LSTMFineTuning(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = SimpleLSTM(config)
        self.add_od_labels=config.add_od_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier=nn.Linear(config.hidden_size, config.num_labels)
        self.config=config

    def forward(self, input_ids, img_feats, labels):
        _, pooled_output=self.lstm(input_ids, img_feats=img_feats) # sequence_output, pooled_output
        # pooled_output = self.dropout(pooled_output)
        logits=self.classifier(pooled_output) # (batch_size, num_labels)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss, logits

class RetrievalDataset(Dataset):
    def __init__(self, tokenizer, args, split: str, is_train: bool):
        super(RetrievalDataset, self).__init__()
        self.data_dir=os.path.join(args.data_dir, split)
        self.img_file=os.path.join(self.data_dir, 'features.tsv')
        self.img_tsv=TSVFile(self.img_file)
        caption_file=os.path.join(self.data_dir, '{}_captions.pt'.format(split))
        self.captions=torch.load(caption_file) # {img_id(int): caption(str)}
        self.img_keys=list(self.captions.keys())

        imgid2idx_file=os.path.join(self.data_dir, 'imageid2idx.json')
        self.image_id2idx=json.load(open(imgid2idx_file)) # {img_id(str): idx(int)}

        if args.add_od_labels:
            label_file=os.path.join(self.data_dir, 'predictions.tsv')
            self.label_tsv=TSVFile(label_file)
            self.labels=dict() # {int: info}
            for line_no in range(self.label_tsv.num_rows()):
                row=self.label_tsv.seek(line_no)
                image_id=row[0] # str
                if int(image_id) in self.img_keys:
                    results=json.loads(row[1]) # dict
                    objects=results['objects'] if type(results) == dict else results # list of dicts
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
        self.is_train=is_train
        self.tokenizer=tokenizer
        self.max_seq_len=args.max_seq_length
        self.max_img_seq_len = args.max_img_seq_length
        self.args=args
    
    def get_od_labels(self, img_key: int):
        if self.args.add_od_labels:
            return ' '.join(self.labels[img_key]['class'])
        else:
            return None
    
    def tensorize_example(self, text_a: str, img_feat, text_b=None, cls_token_segment_id=0, pad_token_segment_id=0, sequence_a_segment_id=0, sequence_b_segment_id=1):
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
        image_idx=self.image_id2idx[str(image_id)]
        row=self.img_tsv.seek(image_idx)
        num_boxes=int(row[1])
        try:
            features=np.frombuffer(base64.b64decode(row[-1]), dtype=np.float32).reshape((num_boxes, -1))
        except ValueError:
            features=np.random.rand(num_boxes, self.args.img_feature_dim).astype('float32')
        t_features=torch.from_numpy(features)
        return t_features
    
    def __len__(self):
        return len(self.img_keys)*self.num_captions_per_img

    def __getitem__(self, index):
        if self.is_train:
            img_key=self.img_keys[index]
            img_feat=self.get_image(img_key)
            caption=self.captions[img_key]
            od_labels=self.get_od_labels(img_key)
            example=self.tensorize_example(text_a=caption, img_feat=img_feat, text_b=od_labels)

            # negative sampling
            neg_img_indexs = list(range(index)) + list(range(index + 1, len(self.img_keys)))
            img_idx_neg=random.choice(neg_img_indexs)
            if random.random()<0.5: # caption from a different image.
                caption_neg = self.captions[self.img_keys[img_idx_neg]]
                example_neg=self.tensorize_example(text_a=caption_neg, img_feat=img_feat, text_b=od_labels)
            else: # randomly select a negative image
                feature_neg = self.get_image(self.img_keys[img_idx_neg])
                od_labels_neg = self.get_od_labels(self.img_keys[img_idx_neg])
                example_neg=self.tensorize_example(text_a=caption, img_feat=feature_neg, text_b=od_labels_neg)
            
            example_pair=tuple(list(example) + [1] + list(example_neg) + [0])
            return index, example_pair
        else:
            img_key = self.img_keys[index]
            feature = self.get_image(img_key)
            caption = self.captions[img_key]
            od_labels = self.get_od_labels(img_key)
            example = self.tensorize_example(text_a=caption, img_feat=feature, text_b=od_labels)

            random_num=random.random()
            if random_num<0.5: # positive example
                label=1
                return index, tuple(list(example) + [label])
            else:
                label=0
                neg_img_indexs = list(range(index)) + list(range(index + 1, len(self.img_keys)))
                img_idx_neg = random.choice(neg_img_indexs)
                if random_num <= 0.75: # negative caption
                    caption_neg = self.captions[self.img_keys[img_idx_neg]]
                    example_neg = self.tensorize_example(text_a=caption_neg, img_feat=feature, text_b=od_labels)
                    return index, tuple(list(example_neg) + [label])
                else: # negative image
                    feature_neg = self.get_image(self.img_keys[img_idx_neg])
                    od_labels_neg = self.get_od_labels(self.img_keys[img_idx_neg])
                    example_neg = self.tensorize_example(text_a=caption, img_feat=feature_neg, text_b=od_labels_neg)
                    return index, tuple(list(example_neg) + [label])

def compute_score_with_logits(logits, labels):
    assert logits.shape[1]==2
    pred=torch.max(logits, dim=1)[1].data
    scores=pred==labels
    c_matrix=confusion_matrix(labels.detach().cpu().numpy(), pred.detach().cpu().numpy(), labels=[0,1])
    TP=c_matrix[1,1]
    TN=c_matrix[0,0]
    FP=c_matrix[0,1]
    FN=c_matrix[1,0]
    return scores, TP, TN, FP, FN

def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir=os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(epoch, global_step))
    mkdir(checkpoint_dir)
    model_to_save=model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, 'pytorch_model.bin'))
    torch.save(args, os.path.join(checkpoint_dir, 'training_args.bin'))
    tokenizer.save_pretrained(checkpoint_dir)
    logger.info("Saving model checkpoint to %s", checkpoint_dir)

def evaluate(args, model, dataloader, length: int):
    model.eval()
    softmax=nn.Softmax(dim=1)
    total_scores=0.0
    total_TP, total_TN, total_FP, total_FN=0,0,0,0
    for _, batch in tqdm(dataloader):
        batch=tuple(t.to(args.device) for t in batch)
        labels=batch[4]
        with torch.no_grad():
            inputs={
                'input_ids': batch[0],
                'img_feats': batch[3],
                'labels': labels
            }
            _, logits=model(**inputs)
            if args.num_labels == 2:
                result = softmax(logits)
            else:
                result=logits
            scores, TP, TN, FP, FN=compute_score_with_logits(result, labels)
            total_scores+=scores.sum().item()
            total_TP+=TP
            total_TN+=TN
            total_FP+=FP
            total_FN+=FN
    precision=total_TP/(total_TP+total_FP)
    recall=total_TP/(total_TP+total_FN)
    return total_scores/length, precision, recall # accuracy, precision, recall

def restore_training_settings(args):
    assert (not args.do_train) and args.do_test
    train_args=torch.load(os.path.join(args.eval_model_dir, 'training_args.bin'))
    override_params=['do_lower_case', 'max_seq_length', 'max_img_seq_length']
    for param in override_params:
        if hasattr(train_args, param):
            train_v=getattr(train_args, param)
            test_v=getattr(args, param)
            if train_v!=test_v:
                logger.warning("Overriding %s: %s -> %s", param, test_v, train_v)
                setattr(args, param, train_v)
    return args

def train(args, train_dataset, val_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader=DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers)
    
    if args.max_steps>0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total=len(train_dataloader)//args.gradient_accumulation_steps*args.num_train_epochs

    if val_dataset is not None:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        val_sampler=SequentialSampler(val_dataset)
        val_dataloader=DataLoader(val_dataset, sampler=val_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)
    
    tb_log_dir=os.path.join(args.output_dir, 'train_logs')
    writer=SummaryWriter(tb_log_dir)

    # no_decay = ['bias', 'LayerNorm.weight']
    # grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay}, # model.named_parameters(): [name (str), value (tensor)]
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer = AdamW(params=grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon) # params (iterable): an iterable of :class:'torch.Tensor's or :class:'dict's. Specifies what Tensors should be optimized
    # if args.scheduler == "constant":
    #     scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps) # Linearly increases lr from 0 to 1 over 'warmup_steps' training steps. Keep lr schedule equal to 1 after warmup_steps
    # elif args.scheduler == "linear":
    #     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total) # Linearly increases lr from 0 to 1 over 'warmup_steps' training steps. Linearly decreases lr from 1 to 0 over remaining t_total-warmup_steps steps
    # else:
    #     raise ValueError("Unknown scheduler {}".format(args.scheduler))

    optimizer=torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: max(1/(1+0.005*epoch), args.min_factor))
    
    if args.n_gpu>1:
        model=nn.DataParallel(model)

    logger.info("***** Running training -- RNN *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc=0, 0., 0.
    model.zero_grad()
    log_json = []
    for epoch in range(args.num_train_epochs):
        for step, (_, batch) in enumerate(train_dataloader):
            model.train()
            batch=tuple(t.to(args.device) for t in batch)
            inputs={
                'input_ids': torch.cat((batch[0], batch[5]), dim=0),
                'img_feats': torch.cat((batch[3], batch[8]), dim=0),
                'labels': torch.cat((batch[4], batch[9]), dim=0)
            }
            loss, logits=model(**inputs)
            if args.n_gpu>1:
                loss=loss.mean()
            if args.gradient_accumulation_steps>1:
                loss=loss/args.gradient_accumulation_steps
            loss.backward() # computes gradient for every parameter
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            batch_score=compute_score_with_logits(logits, inputs['labels'])[0].sum()
            batch_acc=batch_score.item()/(args.train_batch_size*2)
            global_loss+=loss.item()
            global_acc+=batch_acc

            if (step+1)%args.gradient_accumulation_steps==0:
                global_step+=1
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                writer.add_scalar('global loss', global_loss/global_step, global_step)
                writer.add_scalar('global acc', global_acc/global_step, global_step)

                if global_step%args.logging_steps==0:
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
                        val_acc, val_precision, val_recall = evaluate(args, model, val_dataloader, len(val_dataset))
                        epoch_log = {
                            'epoch': epoch, 
                            'global_step': global_step, 
                            'validation acuracy': val_acc
                        }
                        logger.info(
                            'Validation Result: epoch: {}, global_step: {}, validation accuracy: {:.2%}, validation precision: {:.2f}, validation recall: {:.2f}'.format(
                                epoch, global_step, val_acc, val_precision, val_recall
                            )
                        )
                        log_json.append(epoch_log)
                        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                            json.dump(log_json, fp)
                        
                        if len(log_json)>=2:
                            is_imporved=log_json[-1]['validation acuracy']>=log_json[-2]['validation acuracy']
                            if not is_imporved:
                                args.patience-=1
                            if args.patience<=0:
                                logger.info('Early Stopping due to no improvement in validation accuracy')
                                writer.close()
                                return global_step, global_loss/global_step
    writer.close()
    return global_step, global_loss / global_step

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--img_feature_type", default='faster_rcnn', type=str, help="faster_rcnn or mask_rcnn")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False, help="Path to pre-trained model or model type. required for training.")
    parser.add_argument('--data_source', type=str, required=True, help='data source, either coco or wild_data')
    parser.add_argument('--output_dir', type=str, default='relation_output/', help='The output directory to save checkpoint and test results.')
    parser.add_argument("--loss_type", default='sfmx', type=str, help="Loss function types: support kl, sfmx")
    parser.add_argument('--max_seq_length', type=int, default=50, help='max sequence length')
    parser.add_argument('--max_img_seq_length', type=int, default=36, help='max image sequence length')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension, features+location.")
    parser.add_argument('--add_od_labels', default=False, action='store_true', help='add object detection labels')
    parser.add_argument('--do_lower_case', action='store_true', help='do lower case')
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--use_img_layernorm", type=int, default=1, help="Normalize image features with bertlayernorm")
    parser.add_argument("--img_layer_norm_eps", default=1e-12, type=float, help="The epsilon in image feature layer normalization")
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=8)
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=8)
    parser.add_argument("--num_labels", default=2, type=int, help="2 for classification.")
    parser.add_argument("--num_captions_per_img_train", default=1, type=int, help="number of positive matched captions for each training image.")
    parser.add_argument("--num_captions_per_img_val", default=1, type=int, help="number of captions for each validation image.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument("--learning_rate", default=5e-3, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.") # L2 regularization
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.") # clip the gradients by multiplying the unit vector of the gradients with the threshold
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument('--min_factor', type=float, default=0.05, help='minimum factor to be multiplied by learning rate')
    parser.add_argument("--scheduler", default='constant', type=str, help="constant or linear.")
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument("--num_workers", default=6, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1, help="Save checkpoint every X steps. Will also perform validation.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Run evaluation during training at each save_steps.")
    parser.add_argument("--eval_model_dir", type=str, default='', help="Model directory for evaluation.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization.")

    args=parser.parse_args()

    args.data_dir=os.path.join(*['data', args.data_source])

    assert (args.do_train)^(args.do_test), "do_train and do_test must be set exclusively."
    if args.do_train:
        args.output_dir=os.path.join(args.output_dir, os.path.normpath(args.model_name_or_path).split(os.sep)[1])
        mkdir(args.output_dir)

    global logger
    logger=logging.getLogger(__name__)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - [%(name)s - %(filename)s:%(lineno)d] - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    
    args.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.n_gpu=torch.cuda.device_count()
    logger.warning('Device: {}, n_gpu: {}'.format(args.device, args.n_gpu))

    config_class, tokenizer_class = BertConfig, BertTokenizer
    if args.do_train:
        bert='bert-base-multilingual-cased' if 'bert-base-multilingual-cased' in args.model_name_or_path else 'KB/bert-base-swedish-cased'
        config = config_class.from_pretrained(bert, num_labels=args.num_labels, finetuning_task='ir')
        tokenizer=tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_dim=args.img_feature_dim
        config.hidden_dropout_prob=args.drop_out
        config.add_od_labels=args.add_od_labels
        model=LSTMFineTuning(config)
        model_dict=model.state_dict()
        pretrained_dict=torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'), map_location='cpu')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
        model.to(args.device)

        logger.info('Training parameters: {}'.format(args))
        train_dataset=RetrievalDataset(tokenizer=tokenizer, args=args, split='train', is_train=True)
        if args.evaluate_during_training:
            val_dataset=RetrievalDataset(tokenizer=tokenizer, args=args, split='val', is_train=False)
        else:
            val_dataset=None
        global_step, avg_loss=train(args=args, train_dataset=train_dataset, val_dataset=val_dataset, model=model, tokenizer=tokenizer)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)
    else:
        checkpoint=args.eval_model_dir
        assert os.path.isdir(checkpoint)
        bert='bert-base-multilingual-cased' if 'bert-base-multilingual-cased' in checkpoint else 'KB/bert-base-swedish-cased'
        config = config_class.from_pretrained(bert, num_labels=args.num_labels, finetuning_task='ir')
        config.img_feature_dim=args.img_feature_dim
        config.hidden_dropout_prob=args.drop_out
        config.add_od_labels=args.add_od_labels
        tokenizer=tokenizer_class.from_pretrained(checkpoint)
        logger.info('Evaluate the following checkpoint: {}'.format(checkpoint))
        model=LSTMFineTuning(config).to(args.device)
        model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))
        model.to(args.device)
        logger.info('Evaluating parameters: {}'.format(args))
        restore_training_settings(args=args)
        test_dataset=RetrievalDataset(tokenizer=tokenizer, args=args, split='test', is_train=False)
        args.test_batch_size=args.per_gpu_eval_batch_size*max(1, args.n_gpu)
        test_sampler=SequentialSampler(test_dataset)
        test_dataloader=DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size,num_workers=args.num_workers)
        test_acc, test_precision, test_recall=evaluate(args=args, model=model, dataloader=test_dataloader, length=len(test_dataset))
        logger.info('Testing Accuracy: {:.2%}, Precision: {:.2f}, Recall: {:.2f}'.format(test_acc, test_precision, test_recall))
        logger.info('Testing done')

if __name__=='__main__':
    main()