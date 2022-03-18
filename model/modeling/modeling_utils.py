import os
import logging
import torch
import torch.nn.functional as F

from pytorch_transformers.modeling_bert import BertConfig, load_tf_weights_in_bert, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BertPreTrainedModel
from pytorch_transformers.modeling_utils import PreTrainedModel, WEIGHTS_NAME, TF_WEIGHTS_NAME
from pytorch_transformers.file_utils import cached_path

logger=logging.getLogger()

class ImgPreTrainedModel(PreTrainedModel): # no config in PreTrainedModel, but there is one in self defined PreTrainedModel
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config=kwargs.pop('config', None)
        state_dict=kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Load config
        if config is None:
            config, model_kwargs=cls.config_class.from_pretrained(pretrained_model_name_or_path, *model_args, cache_dir=cache_dir, return_unused_kwargs=True, **kwargs)
        else:
            model_kwargs = kwargs
        
        # Load model
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map: # ???
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            if from_tf: # Directly load from a TensorFlow checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
            else:
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        else:
            if from_tf: # Directly load from a TensorFlow checkpoint
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = pretrained_model_name_or_path
        
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                logger.error("Couldn't reach server at '{}' to download pretrained weights.".format(archive_file))
            else:
                logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url.".format(pretrained_model_name_or_path, ', '.join(cls.pretrained_model_archive_map.keys()), archive_file))
            return None
        
        if resolved_archive_file == archive_file:
            logger.info("loading weights file {}".format(archive_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
        
        # Instantiate model
        model = cls(config, *model_args, **model_kwargs)
        
        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')
        
        if from_tf: # Directly load from a TensorFlow checkpoint
            return cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
        
        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        
        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata
        
        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        
        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ''
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            start_prefix = cls.base_model_prefix + '.'
        if hasattr(model, cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            model_to_load = getattr(model, cls.base_model_prefix)
        
        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) == 2 and "size mismatch for cls.seq_relationship.weight" in error_msgs[0]:
            logger.info('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, "\n\t".join(error_msgs)))
        elif len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, "\n\t".join(error_msgs)))
        
        if hasattr(model, 'tie_weights'):
            model.tie_weights()  # make sure word embedding weights are still tied
        
        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model