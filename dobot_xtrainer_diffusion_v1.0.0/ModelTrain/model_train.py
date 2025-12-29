import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
robomimic_path = os.path.join(BASE_DIR, 'robomimic')
print(robomimic_path)
sys.path.append(robomimic_path)
import argparse
from ModelTrain.module.train_module import train
import sys

def arg_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', default='./ckpt/test1',required=False)
    parser.add_argument('--task_name', action='store', type=str, default='demo_test1',help='task_name', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=32, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0,required=False)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', default=50000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=2e-5,required=False)
    parser.add_argument('--load_pretrain', action='store_true', default=False)  # Ignore this parameter and leave the default setting
    parser.add_argument('--eval_every', action='store', type=int, default=1000, help='eval_every', required=False)  # Ignore this parameter and leave the default setting
    parser.add_argument('--validate_every', action='store', type=int, default=1000, help='validate_every',required=False)
    parser.add_argument('--save_every', action='store', type=int, default=5000, help='save_every', required=False)
    # if you want to enhance based on previous checkpoint, use resume ckpt path and set defalut to the path of your checkpoint
    # parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', default= "/home/zz/User_dobot/dobot_xtrainer_yl/ckpt/test1/policy_step_1000_seed_0.ckpt",required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--kl_weight', action='store', type=int, help='KL divergence weight,recommended set 10 or 100', default=10,required=False)
    # 使用diffusion的时候chunk size要为偶数 chunksize shoule be even number when using diffusion policy
    parser.add_argument('--chunk_size', action='store', type=int, help='The model predicts the length of the output action sequence at a time', default=48,required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200,required=False)
    parser.add_argument('--temporal_agg', action='store_true', default=True)
    parser.add_argument('--no_encoder', action='store_true', default=False)

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = arg_config()
    train(args)