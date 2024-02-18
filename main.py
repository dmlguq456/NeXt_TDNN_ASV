import sys, argparse, os
import importlib
import zipfile
import datetime
from collections import OrderedDict
import glob

import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import lightning as L
from lightning.pytorch import seed_everything

from SpeakerNet import SpeakerNet, SpeakerNetMultipleLoss

from engine import ENGINE, ENGINESTEP

from util import *


def train(config, code_save_time):
    # seed
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    seed_everything(config.SEED, workers=True) if hasattr(config, 'SEED') else seed_everything(42, workers=True)

    # ========================================================
    # ⚡⚡ 1. Set 'Dataset', 'DataLoader'  
    # ========================================================
    train_dataset = importlib.import_module('data.dataset').__getattribute__(config.TRAIN_DATASET)
    train_dataset = train_dataset(**config.TRAIN_DATASET_CONFIG) # make instant
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCHSIZE,
        num_workers=config.NUMWORKER,
        pin_memory=True, 
        drop_last=True,
        shuffle=True,
    )

    test_dataset = importlib.import_module('data.dataset').__getattribute__(config.TEST_DATASET)
    test_dataset = test_dataset(**config.TEST_DATASET_CONFIG)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1, 
                                               shuffle=False,
                                               num_workers=config.NUMWORKER,
                                               pin_memory=True, 
                                               drop_last=False)

    # ==============================================================
    # ⚡⚡ 2. Set 'feature_extractor', 'spec_aug', 'model', 'aggregation', 'Loss', 'Optimizer', 'Scheduler'
    # ==============================================================
    # set feature extractor
    feature_extractor = importlib.import_module('preprocessing.' + config.FEATURE_EXTRACTOR).__getattribute__("feature_extractor")
    feature_extractor = feature_extractor(**config.FEATURE_EXTRACTOR_CONFIG)

    # set spec_aug
    spec_aug = importlib.import_module('preprocessing.' + config.SPEC_AUG).__getattribute__("spec_aug")
    spec_aug = spec_aug(**config.SPEC_AUG_CONFIG)

    # set speaker embedding extractor
    model = importlib.import_module('models.' + config.MODEL).__getattribute__("MainModel")
    model =  model(**config.MODEL_CONFIG)

    # set aggregation
    aggregation = importlib.import_module('aggregation.' + config.AGGREGATION).__getattribute__("Aggregation")
    aggregation = aggregation(**config.AGGREGATION_CONFIG)

    # set loss
    loss_function = importlib.import_module("loss." + config.LOSS).__getattribute__("LossFunction")
    loss_function = loss_function(**config.LOSS_CONFIG)

    # mix feature extractor, spec_aug, speaker_verification model & loss function
    speaker_net = SpeakerNet(feature_extractor = feature_extractor,
                       spec_aug = spec_aug, 
                       model = model,
                       aggregation=aggregation,
                       loss_function = loss_function)

    # check model param, mmac
    get_model_param_mmac(speaker_net, int(160*config.EVAL_FRAMES + 240))

    optimizer = importlib.import_module("optimizer." + config.OPTIMIZER).__getattribute__("Optimizer")
    optimizer = optimizer(speaker_net.parameters(), **config.OPTIMIZER_CONFIG)    

    scheduler = importlib.import_module("scheduler." + config.SCHEDULER).__getattribute__("Scheduler")
    scheduler = scheduler(optimizer, **config.SCHEDULER_CONFIG)
    if config.SCHEDULER == "steplr":
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
    elif config.SCHEDULER == "cyclical_lr":
        lr_monitor = LearningRateMonitor(logging_interval='step')
    else:
        raise ValueError("Scheduler is not supported. Please check 'scheduler' in config.py")

    # Load pre-trained weights from checkpoint
    if config.PRETRAINED_CHECKPOINT is not None:
        engine_state_dict = torch.load(config.PRETRAINED_CHECKPOINT)
        # restore model_state_dict
        model_statd_dict = OrderedDict()

        # remove 'model.' from keys in the engine_state_dict
        for k in engine_state_dict['state_dict'].keys():
            # pass loss function
            if 'loss_function' in k:
                if config.RESTORE_LOSS_FUNCTION:
                    model_statd_dict[k.replace('speaker_net.', '')] = engine_state_dict['state_dict'][k]
                else:
                    continue
            
            model_statd_dict[k.replace('speaker_net.', '')] = engine_state_dict['state_dict'][k]

        load_msg = speaker_net.load_state_dict(model_statd_dict, strict=False)
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        print(f"pretrained checkpoint: {config.PRETRAINED_CHECKPOINT}")
        print(load_msg)
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")
    
    # ==============================================================
    # ⚡⚡  3. Set 'engine' for training/validation and 'Trainer'
    # ==============================================================    
    if config.SCHEDULER == "steplr":
        engine = ENGINE(speaker_net = speaker_net,
                        loss_function = speaker_net.loss_function,
                        optimizer = optimizer, 
                        scheduler = scheduler,
                        code_save_time=code_save_time,
                        **config.ENGINE_CONFIG)
    elif config.SCHEDULER == "cyclical_lr":
        engine = ENGINESTEP(speaker_net = speaker_net,
                        loss_function = speaker_net.loss_function,
                        optimizer = optimizer, 
                        scheduler = scheduler,
                        code_save_time=code_save_time,
                        **config.ENGINE_CONFIG)
        
    #  Init ModelCheckpoint callback, monitoring "eer"    
    checkpoint_callback = ModelCheckpoint(**config.CHECKPOINT_CONFIG)

    # LightningModule
    trainer = L.Trainer(
        deterministic=True, # Might make your system slower, but ensures reproducibility.
        default_root_dir = config.TRAINER_CONFIG.get('default_root_dir'), #
        devices = config.TRAINER_CONFIG.get('devices'), #
        val_check_interval = config.TRAINER_CONFIG.get('val_check_interval'), # Check val every n train epochs.
        max_epochs = config.TRAINER_CONFIG.get('max_epochs'), #
        sync_batchnorm = True, # 
        callbacks = [checkpoint_callback, lr_monitor], #
        accelerator = config.TRAINER_CONFIG.get('accelerator'), #
        num_sanity_val_steps = config.TRAINER_CONFIG.get('num_sanity_val_steps'),  # Sanity check runs n batches of val before starting the training routine. This catches any bugs in your validation without having to wait for the first validation check. 
        gradient_clip_val=config.TRAINER_CONFIG.get('gradient_clip_val'), # ⚡⚡
        profiler = config.TRAINER_CONFIG.get('profiler'), #
    )

    # ==============================================================
    # ⚡⚡ 4. run/resume training
    # ==============================================================
    if config.RESUME_CHECKPOINT is not None:
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        print(config.RESUME_CHECKPOINT + "are loaded")
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        trainer.fit(engine, train_loader, test_loader, ckpt_path=config.RESUME_CHECKPOINT)
        
    else:
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        print("resume checkpoint is None")
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        trainer.fit(engine, train_loader, test_loader)

    return True


def test(config):
    # ========================================================
    # ⚡⚡ 1. Set 'Dataset', 'DataLoader'  
    # ========================================================
    test_dataset = importlib.import_module('data.dataset').__getattribute__(config.TEST_DATASET)
    test_dataset = test_dataset(**config.TEST_DATASET_CONFIG)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1, 
                                               shuffle=False,
                                               num_workers=config.NUMWORKER,
                                               pin_memory=True, 
                                               drop_last=False)
    
    # ==============================================================
    # ⚡⚡ 2. Set 'feature_extractor', 'model', 'aggregation'
    # ==============================================================
    # set feature extractor
    feature_extractor = importlib.import_module('preprocessing.' + config.FEATURE_EXTRACTOR).__getattribute__("feature_extractor")
    feature_extractor = feature_extractor(**config.FEATURE_EXTRACTOR_CONFIG)

    # set speaker embedding extractor
    model = importlib.import_module('models.' + config.MODEL).__getattribute__("MainModel")
    model =  model(**config.MODEL_CONFIG)

    # set aggregation
    aggregation = importlib.import_module('aggregation.' + config.AGGREGATION).__getattribute__("Aggregation")
    aggregation = aggregation(**config.AGGREGATION_CONFIG)


    speaker_net = SpeakerNet(feature_extractor = feature_extractor,
                       spec_aug = None, 
                       model = model,
                       aggregation=aggregation,
                       loss_function = None)
    
    # check model param, mmac
    get_model_param_mmac(speaker_net, int(160*config.EVAL_FRAMES + 240))
    
    # =============================================
    # 😀😀 3. load checkpoint
    # =============================================
    """
    if config has 'TEST_CHECKPOINT', load it.
    else, find min eer ckpt from test result(e.g. train/lightning_logs/version_0/checkpoints/###.ckpt) and load it.
    """
    if hasattr(config, 'TEST_CHECKPOINT'):
        min_eer_epoch = config.TEST_CHECKPOINT.split('epoch=')[-1].split('-')[0] # epoch
        engine_state_dict = torch.load(config.TEST_CHECKPOINT)        
    else: # find min eer ckpt
        min_eer_ckpt_path = get_min_eer_ckpt(config)
        min_eer_epoch = min_eer_ckpt_path.split('epoch=')[-1].split('-')[0] # epoch
        engine_state_dict = torch.load(min_eer_ckpt_path)

    # restore model_state_dict
    model_statd_dict = OrderedDict()

    # remove 'model.' from keys in the engine_state_dict
    for k in engine_state_dict['state_dict'].keys():
        model_statd_dict[k.replace('speaker_net.', '')] = engine_state_dict['state_dict'][k]

    load_msg = speaker_net.load_state_dict(model_statd_dict, strict=False)
    print(load_msg)
    # min_eer_epoch = 999
    
    # ==============================================================
    # 😀😀 4. Set 'engine' for training/validation and 'Trainer'
    # ==============================================================
    engine = ENGINE(speaker_net = speaker_net,
                    loss_function = None,
                    optimizer = None, 
                    scheduler = None,
                    code_save_time=datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                    config=config,
                    **config.ENGINE_CONFIG)

    # logger
    csv_logger = CSVLogger(config.TEST_RESULT_PATH, name=f"epoch_{min_eer_epoch}")
    tensorboard_logger = TensorBoardLogger(config.TEST_RESULT_PATH, name=f"epoch_{min_eer_epoch}")

    # GPU    
    trainer = L.Trainer(
        deterministic=False, # Might make your system slower, but ensures reproducibility.
        default_root_dir = config.TEST_RESULT_PATH, #
        devices = config.TRAINER_CONFIG.get('devices'), #
        accelerator = config.TRAINER_CONFIG.get('accelerator'), #
        profiler = config.TRAINER_CONFIG.get('profiler'), #
        logger=[csv_logger, tensorboard_logger],
    )

    trainer.test(engine, test_loader)

    # find best min_eer, min_eer_seg
    find_min_eer_values(config.TRAINER_CONFIG.get('default_root_dir'), ['min_eer', 'min_eer_seg'])

    # epoch
    print(f"epoch_{min_eer_epoch}")

    return True



def test_all(config):
    # ========================================================
    # ⚡⚡ 1. Set 'Dataset', 'DataLoader' for VoxCeleb1-O, VoxCeleb1-E, VoxCeleb1-H
    # ========================================================
    test_dataset = importlib.import_module('data.dataset').__getattribute__('test_dataset')
    # VoxCeleb1-O    
    config.TEST_DATASET_CONFIG['test_list'] = config.TEST_LIST
    test_dataset_O = test_dataset(**config.TEST_DATASET_CONFIG)
    # VoxCeleb1-E
    config.TEST_DATASET_CONFIG['test_list'] = config.TEST_LIST_E # change test_list, TEST_LIST_E: 'list_test_all2.txt'
    test_dataset_E = test_dataset(**config.TEST_DATASET_CONFIG)
    # VoxCeleb1-H
    config.TEST_DATASET_CONFIG['test_list'] = config.TEST_LIST_H # change test_list, TEST_LIST_H: 'list_test_hard2.txt'
    test_dataset_H = test_dataset(**config.TEST_DATASET_CONFIG)
    # test_loader 
    test_loader_O = torch.utils.data.DataLoader(test_dataset_O,
                                               batch_size=1, 
                                               shuffle=False,
                                               num_workers=config.NUMWORKER,
                                               pin_memory=True, 
                                               drop_last=False)
    test_loader_E = torch.utils.data.DataLoader(test_dataset_E,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=config.NUMWORKER,
                                                pin_memory=True,
                                                drop_last=False)
    test_loader_H = torch.utils.data.DataLoader(test_dataset_H,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=config.NUMWORKER,
                                                pin_memory=True,
                                                drop_last=False)
    
    # test_dataloader_dict
    test_dataloader_dict = {'O': test_loader_O, 'E': test_loader_E, 'H': test_loader_H}
    test_list_dict = {'O': config.TEST_LIST, 'E': config.TEST_LIST_E, 'H': config.TEST_LIST_H}
    
    # ==============================================================
    # ⚡⚡ 2. Set 'feature_extractor', 'model', 'aggregation'
    # ==============================================================
    # set feature extractor
    feature_extractor = importlib.import_module('preprocessing.' + config.FEATURE_EXTRACTOR).__getattribute__("feature_extractor")
    feature_extractor = feature_extractor(**config.FEATURE_EXTRACTOR_CONFIG)

    # set speaker embedding extractor
    model = importlib.import_module('models.' + config.MODEL).__getattribute__("MainModel")
    model =  model(**config.MODEL_CONFIG)

    # set aggregation
    aggregation = importlib.import_module('aggregation.' + config.AGGREGATION).__getattribute__("Aggregation")
    aggregation = aggregation(**config.AGGREGATION_CONFIG)


    speaker_net = SpeakerNet(feature_extractor = feature_extractor,
                       spec_aug = None, 
                       model = model,
                       aggregation=aggregation,
                       loss_function = None)
    
    # check model param, mmac
    get_model_param_mmac(speaker_net, int(160*config.EVAL_FRAMES + 240))
    
    # =============================================
    # 😀😀 3. load checkpoint
    # =============================================
    """
    if config has 'TEST_CHECKPOINT', load it.
    else, find min eer ckpt from test result(e.g. train/lightning_logs/version_0/checkpoints/###.ckpt) and load it.
    """
    if hasattr(config, 'TEST_CHECKPOINT'):
        min_eer_epoch = config.TEST_CHECKPOINT.split('epoch=')[-1].split('-')[0] # epoch
        engine_state_dict = torch.load(config.TEST_CHECKPOINT)        
    else: # find min eer ckpt
        min_eer_ckpt_path = get_min_eer_ckpt(config)
        min_eer_epoch = min_eer_ckpt_path.split('epoch=')[-1].split('-')[0] # epoch
        engine_state_dict = torch.load(min_eer_ckpt_path)

    # restore model_state_dict
    model_statd_dict = OrderedDict()

    # remove 'model.' from keys in the engine_state_dict
    for k in engine_state_dict['state_dict'].keys():
        model_statd_dict[k.replace('speaker_net.', '')] = engine_state_dict['state_dict'][k]

    load_msg = speaker_net.load_state_dict(model_statd_dict, strict=False)
    print(load_msg)
    
    for dataset_type, test_loader in test_dataloader_dict.items():
        # ==============================================================
        # 😀😀 4. Set 'engine' for training/validation and 'Trainer'
        # ==============================================================
        engine = ENGINE(speaker_net = speaker_net,
                        loss_function = None,
                        optimizer = None,
                        scheduler = None,
                        code_save_time=datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                        config = config,
                        **config.ENGINE_CONFIG)
        
        # ==============================================================
        # 😀😀 5. run testv for VoxCeleb1-O, VoxCeleb1-E, VoxCeleb1-H
        # ==============================================================
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        print(f"test_dataset_type: VoxCeleb1-{dataset_type}") # O, E, H
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")

        # set evaluation configuration
        engine.eval_config['test_list'] = test_list_dict.get(dataset_type) # O: veri_test2.txt, E: list_test_all2.txt, H: list_test_hard2.txt

        # logger
        csv_logger = CSVLogger(f"{config.TEST_RESULT_PATH}/voxceleb1_{dataset_type}", name=f"epoch_{min_eer_epoch}")
        tensorboard_logger = TensorBoardLogger(f"{config.TEST_RESULT_PATH}/voxceleb1_{dataset_type}", name=f"epoch_{min_eer_epoch}")

        # GPU    
        trainer = L.Trainer(
            deterministic=False, # Might make your system slower, but ensures reproducibility.
            default_root_dir = f"{config.TEST_RESULT_PATH}/voxceleb1_{dataset_type}", #
            devices = config.TRAINER_CONFIG.get('devices'), #
            accelerator = config.TRAINER_CONFIG.get('accelerator'), #
            profiler = config.TRAINER_CONFIG.get('profiler'), #
            logger=[csv_logger, tensorboard_logger],
        )

        trainer.test(engine, test_loader)

        # find best min_eer, min_eer_seg
        find_min_eer_values(f"{config.TEST_RESULT_PATH}/voxceleb1_{dataset_type}", ['min_eer', 'min_eer_seg'])

        # epoch
        print(f"epoch_{min_eer_epoch}")

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Speaker Verification")
    parser.add_argument('--config',         type=str,   default='configs.constant',   help='Config file')
    parser.add_argument('--mode',         type=str,   default='train',   help='choose train | test')

    args = parser.parse_args()

    args.config = args.config.replace('/', '.')
    args.config = args.config.replace('.py', '')
    print(args.config)
    
    config = importlib.import_module(args.config)

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())

    # print arguments, configuration
    print(args)
    logging_terminal(config)

    # ============================================
    # ⚡⚡ Save training code and params - only in training mode
    # ============================================
    if args.mode == "train":
        pyfiles_1st_layer = glob.glob('./*.py')
        pyfiles_2nd_layer = glob.glob('./*/*.py')
        pyfiles = pyfiles_1st_layer + pyfiles_2nd_layer

        code_save_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = f"{config.TRAINER_CONFIG.get('default_root_dir')}/{args.config.split('.')[-1]}_{code_save_time}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_zip_file = save_path + "/code.zip"

        zipf = zipfile.ZipFile(save_zip_file, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)

    # ============================================
    # ⚡⚡ run main script - train/test
    # ============================================
    if args.mode == "train":
        train(config, code_save_time)

    # test VoxCeleb1-O without ASNorm
    elif args.mode == "test":
        test(config)

    # test VoxCeleb1-O, VoxCeleb1-E, VoxCeleb1-H without ASNorm
    elif args.mode == "test_all":
        test_all(config)