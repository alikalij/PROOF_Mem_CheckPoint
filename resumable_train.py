#!/usr/bin/env python3
"""
اسکریپت هوشمند برای ادامه آموزش از آخرین checkpoint
"""

import argparse
import json
import logging
from utils.checkpoint_manager import CheckpointManager
from trainer import _train

def resume_training(config_path, checkpoint_dir="./checkpoints"):
    """ادامه آموزش از آخرین checkpoint"""
    
    # بارگذاری کانفیگ
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # ایجاد مدیر checkpoint
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    # پیدا کردن آخرین checkpoint
    last_task = checkpoint_manager.find_latest_checkpoint()
    
    if last_task is None:
        print("No checkpoint found. Starting new training session.")
        return False
    
    print(f"Resuming from task {last_task}")
    config['resume'] = True
    config['checkpoint_dir'] = checkpoint_dir
    
    try:
        _train(config)
        return True
    except Exception as e:
        print(f"Error resuming training: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    resume_training(args.config, args.checkpoint_dir)