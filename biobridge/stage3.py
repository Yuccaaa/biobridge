import os
try:
    from model.opt_flash_attention import replace_opt_attn_with_flash_attn
except ModuleNotFoundError:
    pass
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from data_provider.stage3_dm import Stage3DM
from data_provider.prot_qa_dm import ProtQADM
from model.blip2_stage3 import Blip2Stage3
from model.dist_funs import MyDeepSpeedStrategy
from pathlib import Path
import pytorch_lightning.callbacks as plc


os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)




def main(args):
    pl.seed_everything(args.seed)
    # model
    model = Blip2Stage3.load_from_checkpoint(args.checkpoint_name, strict=False, args=args, map_location='cpu')
    #print(model.blip2.llm_model)
    print(f"loaded init checkpoint from {args.checkpoint_name}")
    # model = Blip2Stage3(args)
    # model.load_from_stage1_checkpoint(args.checkpoint_name)
    # print(f"loaded stage1 model from {args.checkpoint_name}")
    print('total params:', sum(p.numel() for p in model.parameters()))
    
    dm = Stage3DM(args.dataset, args)
   
    dm.init_tokenizer(model.blip2.llm_tokenizer, model.blip2.plm_tokenizer)
    
    test_loader = dm.test_dataloader()

    # 获取第一条数据
    batch = next(iter(test_loader))
    
    model.eval()

    # 如果你用的是 GPU，则转到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    # 推理
    with torch.no_grad():
        output = model.test_step(batch, 0)  # 0 表示 batch_idx
        print(output)
    
    # trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
    # trainer.validate(model, datamodule=dm)
    # callbacks = []
    # ## fixme save only used parameters
    # # callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", every_n_epochs=10, save_top_k=-1))
    # callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
    #                                      filename='{epoch:02d}', 
    #                                      save_last=True, 
    #                                      save_top_k=-1,
    #                                      save_on_train_epoch_end=False))
    # if len(args.devices.split(',')) > 1:
    #     if args.strategy == 'ddp':
    #         find_unused_parameters = (not args.ptm) or (not args.lm)
    #         strategy = strategies.DDPStrategy(start_method='spawn', find_unused_parameters=find_unused_parameters)
    #     elif args.strategy == 'deepspeed':
    #         strategy = MyDeepSpeedStrategy(stage=2)
    #     else:
    #         NotImplementedError()
    # else:
    #     strategy = None
    #     args.devices = eval(args.devices)
    # if args.use_wandb_logger:
    #     Path(f'./all_checkpoints/{args.filename}/wandb').mkdir(parents=True, exist_ok=True)
    #     logger = WandbLogger(project=args.filename, save_dir=f'./all_checkpoints/{args.filename}/')
    # else:
    #     logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    # trainer = Trainer(
    #     accelerator=args.accelerator,
    #     devices=args.devices,
    #     precision=args.precision,
    #     callbacks=callbacks,
    #     strategy=strategy,
    #     logger=logger,
    #     # limit_train_batches=2,
    #     # limit_val_batches=2,
    # )

    trainer.test(model, datamodule=dm)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    parser.add_argument('--strategy', type=str, default='deepspeed')

    # trainer arguments
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--max_epochs', type=int, default=10)
   # parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--enable_flash', action='store_true', default=False)
    parser.add_argument('--use_wandb_logger', action='store_true', default=False)
    parser = Blip2Stage3.add_model_specific_args(parser)  # add model args
    parser = Stage3DM.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.enable_flash:
        replace_opt_attn_with_flash_attn()
    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args

if __name__ == '__main__':
    main(get_args())

