import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel
from pytorch_transformers.modeling_bert import BertEmbeddings, BertSelfAttention, BertAttention, BertEncoder, BertLayer, BertSelfOutput, BertIntermediate, BertOutput,BertPooler, BertLayerNorm, BertPreTrainedModel, BertLMPredictionHead, BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, load_tf_weights_in_bert
from .modeling_utils import ImgPreTrainedModel

logger=logging.getLogger(__name__)

class CaptionBertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for history_state.
    """
    def __init__(self, config) -> None:
        super().__init__(config) # initialized self.query, self.key and self.value
    
    def forward(self, hidden_states, attention_mask, head_mask=None, history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer=self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # Returns a contiguous in memory tensor containing the same data as self tensor
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class CaptionBertAttention(BertAttention):
    """
    Modified from BertAttention to add support for history_state.
    """
    def __init__(self, config) -> None:
        super().__init__(config) # initialized self.self, self.output
        self.self=CaptionBertSelfAttention(config)
        self.output=BertSelfOutput(config) # dense+dropout+layer_norm
    
    def forward(self, input_tensor, attention_mask, head_mask=None, history_state=None):
        self_outputs=self.self(input_tensor, attention_mask, head_mask, history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class CaptionBertLayer(BertLayer):
    """
    Modified from BertLayer to add support for history_state.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self.attention=CaptionBertAttention(config) # attention output
        self.intermediate=BertIntermediate(config) # dense+glue activation
        self.output=BertOutput(config) # dense+dropout+layer_norm
    
    def forward(self, hidden_states, attention_mask, head_mask=None, history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask, history_state)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

class CaptionBertEncoder(BertEncoder):
    """
    Compute the hidden state from bert layers and become encoder.
    Modified from BertEncoder to add support for encoder_history_states.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer=nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask, head_mask=None, encoder_history_states=None):
        all_hidden_states, all_attentions=(), ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            history_state = None if encoder_history_states is None else encoder_history_states[i]
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], history_state)
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs # (hidden_states, all_hidden_states(optional), all_attentions(optional))

# maybe can remove all history_states
class BertImgModel(BertPreTrainedModel):
    """Expand from BertModel to handle image region features as input
    """
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config) # words_embeddings + position_embeddings + token_type_embeddings
        self.encoder=CaptionBertEncoder(config) # output from the encoder
        self.pooler=BertPooler(config) #taking the hidden state corresponding to the first token.

        self.img_dim=config.img_feature_dim
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        if hasattr(config, 'use_img_layernorm'):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None
        
        if config.img_feature_type == 'dis_code':
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_t': # transpose
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_size, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_scale': # scaled
            self.input_embeddings = nn.Linear(config.code_dim, config.code_size, bias=True)
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        else:
            self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if self.use_img_layernorm:
                self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)
        
        self.apply(self.init_weights)

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
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None, img_feats=None, encoder_history_states=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # a dimension of size one inserted at the specified position
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1) # Returns a new view of the self tensor with singleton dimensions expanded to a larger size, -1 means not changing the size of that dimension
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers
        
        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids) # words_embeddings + position_embeddings + token_type_embeddings
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"
        
        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t': # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale': # left scaled
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            else:
                img_embedding_output = self.img_embedding(img_feats)
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)
                
                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)
            
            # concatenate two embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)
        
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask, encoder_history_states=encoder_history_states) # (hidden_states, all_hidden_states(optional), all_attentions(optional))
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs # (sequence_output, pooled_output, all_hidden_states(optional), all_attentions(optional))

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
        if config.img_feature_dim > 0:
            self.bert=BertImgModel(config)
        else:
            self.bert=BertModel(config)
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
        self.predictions = BertLMPredictionHead(config)
        num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2
        self.seq_relationship = nn.Linear(config.hidden_size, num_seq_relations)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output) # prediction_scores: gelu transform+layernorm
        seq_relationship_score = self.seq_relationship(pooled_output) # applies a linear transformation to the incoming data
        return prediction_scores, seq_relationship_score

class BertImgForPreTraining(ImgPreTrainedModel):
    config_class = BertConfig # basic config for bert: vocab size, hidden size, etc.
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP # {model: url}
    load_tf_weights = load_tf_weights_in_bert # Load tf checkpoints in a pytorch model.
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertImgModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2

        self.apply(self.init_weights)
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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, position_ids=None, head_mask=None, img_feats=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, self.num_seq_relations), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs + (masked_lm_loss,)

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)