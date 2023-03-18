import copy
from utils import MODEL_REGISTRY

from .bert_model import BERT
from .multibert_model import MultiBERT
from .gru_model import BiGRUEncoder

def build_model(cfg):
    model_cfg = copy.deepcopy(cfg)
    try:
        model_cfg = model_cfg['model']
    except Exception:
        raise 'should contain {model}'

    model = MODEL_REGISTRY.get(model_cfg['name'])(**model_cfg['args'])

    return model
