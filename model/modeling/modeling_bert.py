import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers.modeling_utils import PreTrainedModel
from pytorch_transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertLayerNorm, BertPreTrainedModel, BertLMPredictionHead, BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, load_tf_weights_in_bert

logger=logging.getLogger(__name__)

class BertImgModel(BertPreTrainedModel):
    """Expand from BertModel to handle image region features as input
    """
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config) # words_embeddings + position_embeddings + token_type_embeddings
        self.encoder=BertEncoder(config) # output from the encoder, takes embeddings and output the hidden states from the last layer
        self.pooler=BertPooler(config) #taking the hidden state corresponding to the first token, use hanh activation

        self.img_dim=config.img_feature_dim
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        if hasattr(config, 'use_img_layernorm'):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None
        
        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)
        
        self.apply(self.init_weights) # apply(fn): Applies fn recursively to every submodule (including children and itself)

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings
    
    def _prune_heads(self, heads_to_prune):
        """ heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None, img_feats=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # a dimension of size one inserted at the specified position
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        
        # attention_mask is 1.0 for positions we want to attend and 0.0 for masked positions, this operation will create a tensor which is 0.0 for positions we want to attend and -10000.0 for masked positions.
        # add it to the raw scores before softmax
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask: keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] and is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1) # Returns a new view of the self tensor with singleton dimensions expanded to a larger size, -1 means not changing the size of that dimension
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers
        
        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids) # (batch_size, seq_len, hidden_size): words_embeddings + position_embeddings + token_type_embeddings
        
        if img_feats is not None:
            img_embedding_output = self.img_embedding(img_feats)
            if self.use_img_layernorm:
                img_embedding_output = self.LayerNorm(img_embedding_output)
            
            # add dropout on image embedding
            img_embedding_output = self.dropout(img_embedding_output) # (batch_size, img_seq_length, hidden_size)
            
            # concatenate two embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1) # (batch_size, seq_len+img_seq_length, hidden_size)
        
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask) # (hidden_states (the last layer), all_hidden_states at each layer (optional), all_attentions at each layer (optional))
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs # (sequence_output (hidden states of the last layer), pooled_output (pooled hidden states of the last layer), all_hidden_states(optional), all_attentions(optional))

def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss

class ImageBertForSequenceClassification(BertPreTrainedModel):
    """Modified from BertForSequenceClassification
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config
        self.bert=BertImgModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): 
                config.cls_hidden_scale = 2
            
            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier=nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier=nn.Linear(config.hidden_size, self.config.num_labels)
        self.apply(self.init_weights)
    
    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, position_ids=None, head_mask=None, img_feats=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits=self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                labels = labels.to(torch.float)
                loss=loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.loss_type == 'kl':
                    loss_fct=torch.nn.KLDivLoss(reduction='batchmean')
                    log_softmax=torch.nn.LogSoftmax(dim=-1)
                    reshaped_logits=logits.contiguous().view(-1, 3129) # 3129??
                    reshaped_logits=log_softmax(reshaped_logits)
                    loss = loss_fct(reshaped_logits, labels.contiguous())
                elif self.loss_type == 'bce':
                    loss = instance_bce_with_logits(logits, labels)
                else: # retrieval, self.loss_type=sfmx
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs=(loss, )+outputs
        return outputs # (loss, logits, all_hidden_states(optional), all_attentions(optional))

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config) # nn.Linear(config.hidden_size, config.vocab_size)
        num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2
        self.seq_relationship = nn.Linear(config.hidden_size, num_seq_relations)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output) # prediction_scores: (sequence_output.shape[0], vocab_size), Masked Token Loss, prob of words at masked position
        seq_relationship_score = self.seq_relationship(pooled_output) # 0 or 1, whether the image corresponds to the caption, Contrastive Loss.
        return prediction_scores, seq_relationship_score

class BertImgForPreTraining(PreTrainedModel):
    config_class = BertConfig # basic config for bert: vocab size, hidden size, etc.
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP # {model: url}
    load_tf_weights = load_tf_weights_in_bert # Load tf checkpoints in a pytorch model.
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertImgModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2

        self.apply(self.init_weights) # initialize weights
        self.tie_weights()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder, self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, position_ids=None, head_mask=None, img_feats=None): # position_ids and head_mask are not used
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats) # seqence_output from the last layer of encoder, pooled sequence_output from the last layer of encoder

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output) # (batch_size, vocab_size), (batch_size, num_seq_relations)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, self.num_seq_relations), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs + (masked_lm_loss,)

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states (opt)), (attentions (opt)), masked_lm_loss