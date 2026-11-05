"""Public API for classical and neural ECG prediction models."""

from spikeattn.models.base_model import BaseModel
from spikeattn.models.baselines.cnn_baseline import CNNBaseline
from spikeattn.models.baselines.rnn_baseline import RNNBaseline
from spikeattn.models.baselines.svm_baseline import SVMBaseline
from spikeattn.models.snn.attention import InterLeadAttention, TemporalAttention
from spikeattn.models.snn.lif import LIFNode
from spikeattn.models.snn.plain_snn import PlainSNN
from spikeattn.models.snn.spiking_conv import SpikingConv1dBlock
from spikeattn.models.snn.st_snn import SpikeAttentionSNN
from spikeattn.models.snn.surrogate import ArctanSurrogate, surrogate_heaviside

__all__ = [
    "BaseModel",
    "SVMBaseline",
    "CNNBaseline",
    "RNNBaseline",
    "PlainSNN",
    "SpikeAttentionSNN",
    "LIFNode",
    "ArctanSurrogate",
    "surrogate_heaviside",
    "TemporalAttention",
    "InterLeadAttention",
    "SpikingConv1dBlock",
]
