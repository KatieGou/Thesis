import json
import logging
import os
import base64
import argparse
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import warnings
warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter
from model.utils.misc import mkdir
from model.utils.tsv_file import TSVFile

from pytorch_transformers import BertTokenizer

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
    
    def tensorize_example(self, text_a: str, img_feat, text_b=None):
        tokens=list()
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens+=tokens_a
        if len(tokens)>self.max_seq_len:
            tokens=tokens[:self.max_seq_len]
        seq_len = len(tokens)

        if seq_len<self.max_seq_len and text_b is not None:
            tokens_b=self.tokenizer.tokenize(text_b)
            if len(tokens_b)>self.max_seq_len-seq_len:
                tokens_b=tokens_b[:self.max_seq_len-seq_len]
            tokens+=tokens_b

        seq_len=len(tokens)
        seq_padding_len=self.max_seq_len-seq_len
        tokens+=[self.tokenizer.pad_token]*seq_padding_len
        input_ids=self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids=torch.unsqueeze(torch.Tensor(input_ids), 0)

        img_len=img_feat.shape[0]
        if img_len>self.max_img_seq_len:
            img_feat=img_feat[:self.max_img_seq_len]
        else:
            img_padding_len=self.max_img_seq_len-img_len
            img_feat=np.concatenate([img_feat, np.zeros((img_padding_len, img_feat.shape[1]), dtype=np.float32)], axis=0)
            img_feat=torch.from_numpy(img_feat)
        
        return torch.cat([input_ids, img_feat], dim=0)
    
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
        return len(self.img_keys)

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
            
            example_pair=tuple([example]+[1]+[example_neg]+[0])
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
                return index, tuple([example] + [label])
            else:
                label=0
                neg_img_indexs = list(range(index)) + list(range(index + 1, len(self.img_keys)))
                img_idx_neg = random.choice(neg_img_indexs)
                if random_num <= 0.75: # negative caption
                    caption_neg = self.captions[self.img_keys[img_idx_neg]]
                    example_neg = self.tensorize_example(text_a=caption_neg, img_feat=feature, text_b=od_labels)
                    return index, tuple([example_neg] + [label])
                else: # negative image
                    feature_neg = self.get_image(self.img_keys[img_idx_neg])
                    od_labels_neg = self.get_od_labels(self.img_keys[img_idx_neg])
                    example_neg = self.tensorize_example(text_a=caption, img_feat=feature_neg, text_b=od_labels_neg)
                    return index, tuple([example_neg] + [label])
    
class Baseline(nn.Module):
    def __init__(self, args):
        super(Baseline, self).__init__()
        self.nn=nn.Sequential(
            nn.Flatten(),
            nn.Linear(args.input_size, 2**10),
            nn.LeakyReLU(),
            nn.Linear(2**10, 2**8),
            nn.LeakyReLU(),
            nn.Linear(2**8, 2**6),
            nn.LeakyReLU(),
            nn.Linear(2**6, 2**4),
            nn.LeakyReLU(),
            nn.Linear(2**4, args.output_size),
        )

    def forward(self, x):
        return self.nn(x)

def compute_score_with_logits(logits, labels):
    assert logits.shape[1]==2
    pred=torch.max(logits, dim=1)[1].data
    scores=pred==labels
    return scores.sum()

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
    for _, batch in tqdm(dataloader):
        batch=tuple(t.to(args.device) for t in batch)
        x=batch[0]
        labels=batch[1]
        with torch.no_grad():
            outputs=model(x)
            logits=softmax(outputs)
            scores=compute_score_with_logits(logits, labels)
            total_scores+=scores.item()
    return total_scores/length # accuracy

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

def train(args, train_dataset, val_dataset, model, tokenizer, loss_fn):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader=DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers)
    
    t_total=len(train_dataloader)//args.gradient_accumulation_steps*args.num_train_epochs

    if val_dataset is not None:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        val_sampler=SequentialSampler(val_dataset)
        val_dataloader=DataLoader(val_dataset, sampler=val_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)
    
    tb_log_dir=os.path.join(args.output_dir, 'train_logs')
    writer=SummaryWriter(tb_log_dir)

    optimizer=torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: max(1/(1+0.005*epoch), args.min_factor))

    logger.info("***** Running training -- BaseLine *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc=0, 0., 0.
    model.zero_grad()
    log_json = []
    for epoch in range(args.num_train_epochs):
        for step, (_, batch) in enumerate(train_dataloader):
            model.train()
            batch=tuple(t.to(args.device) for t in batch)
            x=torch.cat([batch[0], batch[2]], dim=0)
            labels=torch.cat([batch[1], batch[3]], dim=0)
            outputs=model(x)
            loss=loss_fn(outputs, labels)
            if args.n_gpu > 1: 
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward() # computes gradient for every parameter
            logits=nn.Softmax(dim=1)(outputs)
            batch_score=compute_score_with_logits(logits, labels)
            batch_acc=batch_score.item()/(2*args.train_batch_size)
            global_loss+=loss.item()
            global_acc+=batch_acc
            if (step+1)%args.gradient_accumulation_steps==0:
                global_step+=1
                optimizer.step() # apply grad of parameters
                scheduler.step() # update learning rate
                model.zero_grad()
                
                writer.add_scalar('global loss', global_loss/global_step, global_step)
                writer.add_scalar('global acc', global_acc/global_step, global_step)
                
                if global_step%args.logging_steps==0:
                    logger.info(
                        "Epoch: {}, global_step: {}, lr: {:.6f}, batch loss: {:.4f}, global loss: {:.4f}, batch acc: {:.2%}, global acc: {:.2%}".format(
                            epoch, global_step, optimizer.param_groups[0]["lr"], loss, global_loss / global_step, batch_acc, global_acc / global_step
                        )
                    )
                
                if (args.save_steps>0 and global_step%args.save_steps==0) or (global_step==t_total):
                    save_checkpoint(model, tokenizer, args, epoch, global_step)
                    if args.evaluate_during_training:
                        logger.info('Perform evaluation at step {}'.format(global_step))
                        logger.info('Number validation examples = %d', len(val_dataset))
                        logger.info('Evalutation batch size = %d', args.eval_batch_size)
                        val_acc=evaluate(args, model, val_dataloader, len(val_dataset))
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
                        with open(os.path.join(args.output_dir, 'eval_logs.json'), 'w') as f:
                            json.dump(log_json, f)
                        
                        if len(log_json)>=2:
                            is_imporved=log_json[-1]['validation acuracy']>=log_json[-2]['validation acuracy']
                            if not is_imporved:
                                args.patience-=1
                            if args.patience<=0:
                                logger.info('Early Stopping due to no improvement in validation accuracy')
                                writer.close()
                                return global_step, global_loss/global_step
    writer.close()
    return global_step, global_loss/global_step

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--img_feature_type", default='faster_rcnn', type=str, help="faster_rcnn or mask_rcnn")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False, help="Path to pre-trained model or model type. required for training.")
    parser.add_argument('--output_dir', type=str, default='baseline_output/', help='output directory')
    parser.add_argument('--max_seq_length', type=int, default=2054, help='max sequence length')
    parser.add_argument('--max_img_seq_length', type=int, default=30, help='max image sequence length')
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension, features+location.")
    parser.add_argument('--add_od_labels', default=True, help='add object detection labels')
    parser.add_argument('--do_lower_case', action='store_true', help='do lower case')
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_during_training', action='store_true', help="Run evaluation during training at each save_steps.")
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=128)
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--min_factor', type=float, default=0.05, help='minimum factor to be multiplied by learning rate')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=500, help="Save checkpoint every X steps.")
    parser.add_argument('--eval_model_dir', type=str, default='baseline_output/', help="Teting directory containing the saved model.")
    args=parser.parse_args()

    args.data_dir=os.path.join(*['data', args.img_feature_type, 'coco'])

    assert (args.do_train)^(args.do_test), "do_train and do_test must be set exclusively."

    args.input_size=args.img_feature_dim*(args.max_img_seq_length+1)

    global logger
    logger=logging.getLogger(__name__)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - [%(name)s - %(filename)s:%(lineno)d] - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    
    args.output_dir=os.path.join(args.output_dir, os.path.normpath(args.model_name_or_path).split(os.sep)[1])
    mkdir(args.output_dir)

    args.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.n_gpu=torch.cuda.device_count()
    logger.warning('Device: {}, n_gpu: {}'.format(args.device, args.n_gpu))

    tokenizer_class = BertTokenizer
    if args.do_train:
        tokenizer=tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        model=Baseline(args).to(args.device)
        logger.info('Training parameters: {}'.format(args))
        train_dataset=RetrievalDataset(tokenizer=tokenizer, args=args, split='train', is_train=True)
        if args.evaluate_during_training:
            val_dataset=RetrievalDataset(tokenizer=tokenizer, args=args, split='val', is_train=False)
        else:
            val_dataset=None
        loss_fn=nn.CrossEntropyLoss() # perform softmax internally
        global_step, avg_loss=train(args=args, train_dataset=train_dataset, val_dataset=val_dataset, model=model, tokenizer=tokenizer, loss_fn=loss_fn)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)
    else:
        checkpoint=args.eval_model_dir
        assert os.path.isdir(checkpoint)
        tokenizer=tokenizer_class.from_pretrained(checkpoint)
        logger.info('Evaluate the following checkpoint: {}'.format(checkpoint))
        model=Baseline(args).to(args.device)
        model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))
        model.to(args.device)
        logger.info('Evaluating parameters: {}'.format(args))
        restore_training_settings(args=args)
        test_dataset=RetrievalDataset(tokenizer=tokenizer, args=args, split='test', is_train=False)
        args.test_batch_size=args.per_gpu_eval_batch_size*max(1, args.n_gpu)
        test_sampler=SequentialSampler(test_dataset)
        test_dataloader=DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size,num_workers=args.num_workers)
        test_acc=evaluate(args=args, model=model, dataloader=test_dataloader, length=len(test_dataset))
        logger.info("Test accuracy: %s", test_acc)
        logger.info('Testing done')

if __name__=='__main__':
    main()