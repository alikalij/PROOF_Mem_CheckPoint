import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
from utils.checkpoint_manager import CheckpointManager


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args, start_task=0):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(args["model_name"], args["dataset"], 
        init_cls, args["increment"], args["prefix"], args["seed"],args["convnet_type"],)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    
    # ایجاد مدیر checkpoint
    checkpoint_dir = args.get("checkpoint_dir", "/content/drive/MyDrive/saved_models/PROOF_Mem_Checkpoints")
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    data_manager = DataManager(args["dataset"],args["shuffle"],args["seed"],args["init_cls"],args["increment"], )
    model = factory.get_model(args["model_name"], args)
    model.save_dir=logs_name

    # بررسی ادامه از checkpoint
    start_task = 0
    if args.get("resume", False):
        last_task = checkpoint_manager.find_latest_checkpoint()
        if last_task is not None:            
            success = checkpoint_manager.load_checkpoint(model, last_task, args["device"][0])
            if success:
                start_task = last_task + 1
                logging.info(f"Resuming from task {start_task}")
            else:
                logging.warning("Failed to load checkpoint, starting from scratch")
        else:
            logging.info("No checkpoint found, starting from scratch")

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    zs_seen_curve, zs_unseen_curve, zs_harmonic_curve, zs_total_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}, {"top1": [], "top5": []}, {"top1": [], "top5": []}
    logging.info("data_manager.nb_tasks=> {}".format(data_manager.nb_tasks))
    for task in range(start_task, data_manager.nb_tasks):
        try:
            logging.info(f"Starting task {task}")
            logging.info("All params: {}".format(count_parameters(model._network)))
            logging.info(
                "Trainable params: {}".format(count_parameters(model._network, True))
            )
            model.incremental_train(data_manager)
            logging.info(f"log1===========")
            cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total = model.eval_task()
            logging.info(f"log2===========")
            # ذخیره checkpoint
            metrics = {
                'accuracy': cnn_accy["top1"],
                'task': task,
                'total_classes': model._total_classes
            }
            checkpoint_manager.save_checkpoint(model, task, metrics)
            logging.info(f"Saved task {task}")

            model.after_task()
        
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
    
        except Exception as e:
                logging.error(f"Error in task {task}: {e}")
                # ذخیره emergency checkpoint
                checkpoint_manager.save_checkpoint(model, task, {'error': str(e)})
                raise

        logging.info("Training completed successfully!")
        
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
