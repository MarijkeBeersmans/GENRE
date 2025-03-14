import os
import sys
sys.path.append("/home/pricie/marijkeb/github/GENRE/fairseq")
sys.path.append("/home/pricie/marijkeb/github/GENRE")
import torch
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    FlaxBartForConditionalGeneration,
    TFBartForConditionalGeneration,
)

from genre.fairseq_model import mGENRE


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = torch.nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# Load


fairseq_path = "../models/fairseq_multilingual_entity_disambiguation"
hf_path = "../models/hf_multilingual_entity_disambiguation"
# fairseq_path = "../models/fairseq_entity_disambiguation_aidayago"
# hf_path = "../models/hf_entity_disambiguation_aidayago"

# import omegaconf
# import collections
# from typing import Any

# torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
# torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
# torch.serialization.add_safe_globals([dict])
# torch.serialization.add_safe_globals([collections.defaultdict])
# torch.serialization.add_safe_globals([Any])
# torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
# torch.serialization.add_safe_globals([omegaconf.base.Metadata])

fairseq_model = mGENRE.from_pretrained(fairseq_path).eval()
config = BartConfig(vocab_size=256001)
hf_model = BartForConditionalGeneration(config).eval()
hf_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
hf_tokenizer.save_pretrained(hf_path)

# Convert pytorch

state_dict = fairseq_model.model.state_dict()
remove_ignore_keys_(state_dict)
state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
hf_model.model.load_state_dict(state_dict)
hf_model.lm_head = make_linear_from_emb(hf_model.model.shared)
hf_model.save_pretrained(hf_path)

# Convert flax

hf_model = FlaxBartForConditionalGeneration.from_pretrained(hf_path, from_pt=True)
hf_model.save_pretrained(hf_path)

# Convert tensorflow

hf_model = TFBartForConditionalGeneration.from_pretrained(hf_path, from_pt=True)
hf_model.save_pretrained(hf_path)
