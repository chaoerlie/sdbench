import evaluate.IAP.IAP_inference as IAP_inference
from hps_calculator import HPSCalculator

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def evaluate(img_path):

    return IAP_inference.evaluateImg(img_path)


