## Run all dependencies
import os
# # os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'
# import sys
# sys.path.append(os.path.abspath('..'))
# PYTHONPATH="${PYTHONPATH}:/Users/macos/desktop/nc/"
# 配置参数
from bilstm_onefile.components.train import train_save
from bilstm_onefile.components.evaluate import evaluate_model
import warnings
warnings.filterwarnings("ignore")

def run_bilstm():

    train_save()
    evaluate_model()


if __name__ == '__main__':
    run_bilstm()
