"""
LAIGCD Models
"""

from .freq_module import FreqModule, DCTModule, SRMConv
from .prototype import PrototypeModule, SimpleClassifier
from .detector import LightweightAIGCDetector, build_model
from .fakevlm_explainer import FakeVLMExplainer, create_explainer


__all__ = [
    'FreqModule',
    'DCTModule',
    'SRMConv',
    'PrototypeModule',
    'SimpleClassifier',
    'LightweightAIGCDetector',
    'build_model',
    'FakeVLMExplainer',
    'create_explainer',
]
