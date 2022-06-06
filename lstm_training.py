import datetime
import json
import logging
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import warnings
warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter
from model.utils.misc import mkdir
from model.datasets.build import make_data_loader
from model.modeling.modeling_bert import BertPreTrainingHeads

from pytorch_transformers import BertConfig, AdamW, WarmupLinearSchedule

class SimpleLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding=nn.Embedding(config.vocab_size, config.hidden_size)
        self.img_dim=config.img_feature_dim
        self.img_embedding = nn.Linear(self.img_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.lstm=nn.LSTM(config.hidden_size, int(config.hidden_size/2), num_layers=12, bidirectional=True, batch_first=True)
        
    def forward(self, input_ids, img_feats=None):
        embedding_output=self.embedding(input_ids)
        if img_feats is not None:
            img_embedding_output = self.img_embedding(img_feats)
            img_embedding_output = self.dropout(img_embedding_output) # (batch_size, img_seq_length, hidden_size)
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1) # (batch_size, seq_len+img_seq_length, hidden_size)
        lstm_outputs=self.lstm(embedding_output) # tuple of (output, hidden)
        sequence_output, _=lstm_outputs # (batch_size, seq_len, hidden_size)
        first_token_tensor=sequence_output[:,0,:]
        pooled_output=self.dense(first_token_tensor)
        pooled_output=self.activation(pooled_output) # (batch_size, hidden_size)
        outputs=(sequence_output, pooled_output)
        return outputs # sequence_output, pooled_output

class SimpleLSTMPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm=SimpleLSTM(config)
        self.cls=BertPreTrainingHeads(config)
        self.num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2
        self.add_od_labels=config.add_od_labels
        self.config=config
    
    def forward(self, input_ids, masked_lm_labels=None, next_sentence_label=None, img_feats=None):
        outputs=self.lstm(input_ids, img_feats)
        sequence_output, pooled_output=outputs
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output) # (batch_size, vocab_size), (batch_size, num_seq_relations)
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1) # ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, self.num_seq_relations), next_sentence_label.view(-1))
            if self.add_od_labels:
                total_loss = masked_lm_loss + next_sentence_loss
            else:
                total_loss = masked_lm_loss
            outputs = (total_loss,) + outputs
            return outputs

def main():
    logger=logging.getLogger(__name__)

    parser=argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir containing the .yaml files for the task.")
    parser.add_argument("--dataset_file", default=None, type=str, required=True, help="The training dataset yaml file.")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--add_od_labels", default=True, action='store_true', help="Whether to add object detection labels or not.")

    parser.add_argument("--max_img_seq_length", default=36, type=int, help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='faster_rcnn', type=str, help="faster_rcnn or mask_rcnn")

    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out for BERT.")
    parser.add_argument("--use_b", type=int, default=1, help="use text b")
    parser.add_argument("--textb_sample_mode", type=int, default=1, help="0: sample from both texta&textb, 1: sample from textb, 2: sample from QA answers")
    parser.add_argument("--texta_false_prob", type=float, default=0.0, help="the probality that we sample wrong texta, should in [0.0, 0.5]")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or model type. required for training.")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length", default=50, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_iters", default=200, type=int, help="Maximal number of training iterations.")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size for training.")
    parser.add_argument("--num_workers", default=6, type=int, help="Number of workers for dataset.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=-1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--do_lower_case", action='store_true', help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumualte before performing a backward/update pass.")

    parser.add_argument("--use_img_layernorm", type=int, default=0, help="Normalize image features with bertlayernorm")
    parser.add_argument("--img_layer_norm_eps", default=1e-12, type=float, help="The eps in image feature laynorm layer")

    parser.add_argument("--mask_loss_for_unmatched", type=int, default=1, help="masked language model loss for unmatched triplets")
    parser.add_argument("--use_gtlabels", type=int, default=1, help="use groundtruth labels for text b or not")
    
    parser.add_argument('--ckpt_period', type=int, default=100, help="Period for saving checkpoint")
    parser.add_argument('--log_period', type=int, default=20, help="Period for saving logging info")

    args=parser.parse_args()

    args.output_dir=os.path.join(args.output_dir, '_'.join(['lstm', args.model_name_or_path.replace('/', '_')]))   
    
    assert args.do_train, "Training is currently the only implemented execution option. Please set do_train."

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        logger.info("Output Directory Exists.")
    
    args.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu=torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - [%(name)s - %(filename)s:%(lineno)d] - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger.warning(
        "Device: %s, n_gpu: %s, distributed training: %s",
        args.device, args.n_gpu, bool(args.n_gpu>1)
    )

    assert args.gradient_accumulation_steps>=1, "Gradient_accumulation_steps should be >= 1"

    if not os.path.exists(args.output_dir):
        mkdir(args.output_dir)
    
    config_class=BertConfig
    config=config_class.from_pretrained(args.model_name_or_path)
    config.img_layer_norm_eps = args.img_layer_norm_eps
    config.use_img_layernorm = args.use_img_layernorm

    # discrete code
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.hidden_dropout_prob = args.drop_out
    args.num_contrast_classes = 2
    config.num_contrast_classes = args.num_contrast_classes
    config.add_od_labels = args.add_od_labels
    config.max_img_seq_length=args.max_img_seq_length
    config.max_seq_length=args.max_seq_length

    model=SimpleLSTMPretraining(config)
    for key, val in vars(config).items():
        setattr(args, key, val)
    
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    tb_log_dir = os.path.join(args.output_dir, 'train_logs')
    writer=SummaryWriter(tb_log_dir)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters()) # Returns an iterator over module parameters, yielding both the name of the parameter and the parameter itself
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.max_iters) # linearly increases learning rate from 0 to args.learning_rate over args.warmup_steps, linealy dcreases from args.learning_rate to 0 over args.max_iters-args.warmup_steps

    if args.n_gpu>1:
        model=torch.nn.DataParallel(model)
    
    train_dataloaders=make_data_loader(args, is_distributed=False)

    if isinstance(train_dataloaders, list):
        train_dataloader=train_dataloaders[0]
    else:
        train_dataloader=train_dataloaders
    tokenizer = train_dataloader.dataset.tokenizer

    max_iter=len(train_dataloader)
    start_iter=0
    logger.info("***** Running training *****")
    logger.info("  Num examples = {}".format(len(train_dataloader.dataset))) # train_dataloader.dataset=OscarTSVDataset
    logger.info("  Instantaneous batch size = %d", args.train_batch_size // args.gradient_accumulation_steps)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", max_iter//args.gradient_accumulation_steps)

    def data_process(mini_batch):
        images, targets = mini_batch[0], mini_batch[1] # from __getitem__: img feature, img infos, index
        targets_transposed = list(zip(*targets))
        '''
        input_ids: Indices of input sequence tokens in the vocabulary.
        input_mask: Mask to avoid performing attention on padding token indices. 1: no mask, 0: mask (padded).
        segment_ids: token_type_ids. Segment token indices to indicate first and second portions of the inputs, 0: sentence A, 1: sentence B.
        lm_label_ids: indices of masked tokens in the vocabulary (-1 for original tokens)
        is_next: whether textb (tags) is the next sentence of texta (captions)
        '''
        input_ids = torch.stack(targets_transposed[0]).to(args.device, non_blocking=True) # non_blocking=True: copy asynchronously to args.device, should be used when pin_memory=True
        input_mask = torch.stack(targets_transposed[1]).to(args.device, non_blocking=True)
        segment_ids = torch.stack(targets_transposed[2]).to(args.device, non_blocking=True)
        lm_label_ids = torch.stack(targets_transposed[3]).to(args.device, non_blocking=True)
        is_next = torch.stack(targets_transposed[4]).to(args.device, non_blocking=True)
        is_img_match = torch.stack(targets_transposed[5]).to(args.device, non_blocking=True)
        return images, input_ids, input_mask, segment_ids, lm_label_ids, is_next
    
    def forward_backward(images, input_ids, segment_ids, lm_label_ids, is_next, loss_weight=1.0): # feature as input
        image_features = torch.stack(images).to(args.device, non_blocking=True)
        outputs = model(input_ids, masked_lm_labels=lm_label_ids, next_sentence_label=is_next, img_feats=image_features) # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)
        loss = loss_weight * outputs[0]
        if args.n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward() # computes gradient for every parameter
        return loss.item()
    
    if os.path.exists(os.path.join(args.output_dir, 'loss_logs.json')):
        with open(os.path.join(args.output_dir, 'loss_logs.json'), 'r') as f:
            log_json = json.load(f)
    else:
        log_json=dict()
    
    model.train()
    model.zero_grad()

    clock_started = False # Every args.ckpt_period, report train_score and save model
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(train_dataloader, start_iter):
        if not clock_started:
            start_training_time = time.time()
            clock_started = True
        
        images, input_ids, input_mask, segment_ids, lm_label_ids, is_next = data_process(batch)

        loss = forward_backward(images, input_ids, segment_ids, lm_label_ids, is_next, loss_weight=1.0)
        tr_loss += loss
        nb_tr_steps += 1
        global_loss=tr_loss/nb_tr_steps

        writer.add_scalar('global loss', global_loss, step+1)

        if (step+1) % args.gradient_accumulation_steps == 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # Clips gradient norm of an iterable of parameters
            # do the optimization steps
            optimizer.step() # apply grad of parameters
            scheduler.step() # Update learning rate schedule
            model.zero_grad() # optimizer.zero_grad()

            if args.log_period > 0 and (step + 1) % args.log_period == 0:
                logger.info(
                    'Step: {}, lr: {:.7f}, batch_loss: {:.4f}, global_loss: {:.4f}'.format(
                        step+1, optimizer.param_groups[0]["lr"], loss, global_loss
                    )
                )
        
        if (step + 1) == max_iter or (step + 1) % args.ckpt_period == 0:  # Save a trained model
            log_json[step+1] = tr_loss
            logger.info("PROGRESS: {}%".format(round(100 * (step + 1) / max_iter, 4)))
            with open(os.path.join(args.output_dir, 'loss_logs.json'), 'w') as fp:
                json.dump(log_json, fp)
            
            # save checkpoint
            output_dir = os.path.join(args.output_dir,'checkpoint-{:07d}'.format(step + 1))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            tokenizer.save_pretrained(output_dir)
            logger.info( "Saving model checkpoint {0} to {1}".format(step + 1, output_dir))
    
    if clock_started:
        total_training_time = time.time() - start_training_time
    else:
        total_training_time = 0.0
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    writer.close()

if __name__ == "__main__":
    main()