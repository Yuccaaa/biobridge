import argparse
from pathlib import Path
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

if __name__ == '__main__':
    ## read a path using argparse and pass it to convert_zero_checkpoint_to_fp32_state_dict
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='path to the desired checkpoint folder')
    parser.add_argument('--output', type=str, default=None, help='path to the pytorch fp32 state_dict output file')
    # parser.add_argument('--tag', type=str, help='checkpoint tag used as a unique identifier for checkpoint')
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(args.input) / 'converted.ckpt'
    convert_zero_checkpoint_to_fp32_state_dict(args.input, args.output)

# import argparse
# from pathlib import Path
# from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
# from torch.serialization import add_safe_globals
# from deepspeed.runtime.fp16.loss_scaler import LossScaler
# from deepspeed.runtime.zero.config import ZeroStageEnum
# from deepspeed.utils.tensor_fragment import fragment_address
# import torch

# if __name__ == '__main__':
#     # 添加DeepSpeed的LossScaler到安全名单
#     add_safe_globals([LossScaler, ZeroStageEnum])  # 添加ZeroStageEnum

#     torch.serialization.safe_globals([LossScaler])
#     torch.serialization.safe_globals([fragment_address])
#     # 读取路径参数
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', type=str, required=True, help='path to the desired checkpoint folder')
#     parser.add_argument('--output', type=str, default=None, help='path to the pytorch fp32 state_dict output file')
#     args = parser.parse_args()

#     # 设置默认输出路径
#     if args.output is None:
#         args.output = str(Path(args.input).parent / 'converted_fp32.ckpt')

#     # 执行转换
#     try:
#         convert_zero_checkpoint_to_fp32_state_dict(args.input, args.output)
#         print(f"Successfully converted checkpoint to: {args.output}")
#     except Exception as e:
#         print(f"Conversion failed: {str(e)}")