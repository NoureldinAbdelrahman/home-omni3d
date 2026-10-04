import sys
import auxiliary.argument_parser as argument_parser
import auxiliary.my_utils as my_utils
import time
import random
import numpy as np
import torch
from auxiliary.my_utils import yellow_print

"""
Main training script.
author : Thibault Groueix 01.11.2019
"""

opt = argument_parser.parser()
torch.cuda.set_device(opt.multi_gpu[0])
my_utils.plant_seeds(random_seed=opt.random_seed)
if getattr(opt, "seed", -1) >= 0:
    # Manual seed override (e.g. seeds x2 repeats for the winning config).
    # Recorded in options.json like every other flag.
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    print(f"manual seed override: {opt.seed}")
import training.trainer as trainer

trainer = trainer.Trainer(opt)
trainer.build_dataset()
trainer.build_network()
trainer.build_optimizer()
trainer.build_losses()
trainer.start_train_time = time.time()

if opt.demo:
    with torch.no_grad():
        trainer.demo(opt.demo_input_path)
    sys.exit(0)

if opt.run_single_eval:
    with torch.no_grad():
        trainer.test_epoch()
    sys.exit(0)

for epoch in range(trainer.epoch, opt.nepoch):
    trainer.train_epoch()
    with torch.no_grad():
        trainer.test_epoch()
    trainer.dump_stats()
    trainer.increment_epoch()
    trainer.save_network()

yellow_print(f"Visdom url http://localhost:{trainer.opt.visdom_port}/")
yellow_print(f"Netvision report url http://localhost:{trainer.opt.http_port}/{trainer.opt.dir_name}/index.html")
yellow_print(f"Training time {(time.time() - trainer.start_time)//60} minutes.")
