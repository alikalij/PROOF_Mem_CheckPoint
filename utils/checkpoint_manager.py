import os
import torch
import logging
import json
import time
from pathlib import Path

class CheckpointManager:
    def __init__(self, checkpoint_dir="./checkpoints", auto_save=True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.auto_save = auto_save
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, model, task_id, metrics=None, data_manager=None):
        """ذخیره کامل وضعیت آموزش"""
        checkpoint_path = self.checkpoint_dir / f"task_{task_id}.pth"
        
        # جمع‌آوری تمام stateهای مهم
        checkpoint_data = {
            'model_state_dict': model._network.state_dict(),
            'task_id': task_id,
            'known_classes': model._known_classes,
            'total_classes': model._total_classes,
            'cur_task': model._cur_task,
            'global_prototypes': model.global_prototypes if hasattr(model, 'global_prototypes') else None,
            'prototype_memory': model.prototype_memory if hasattr(model, 'prototype_memory') else {},
            'metrics': metrics or {},
            'timestamp': time.time()
        }
        
        # ذخیره وضعیت optimizer و scheduler اگر وجود دارند
        if hasattr(model, 'optimizer'):
            checkpoint_data['optimizer_state_dict'] = model.optimizer.state_dict()
        if hasattr(model, 'scheduler'):
            checkpoint_data['scheduler_state_dict'] = model.scheduler.state_dict()
        
        # ذخیره حافظه replay اگر وجود دارد
        if hasattr(model, '_memory') and model._memory is not None:
            checkpoint_data['memory_data'] = model._memory
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            
            # ذخیره متادیتا
            meta_path = self.checkpoint_dir / "training_meta.json"
            meta_data = {
                'last_task': task_id,
                'total_classes': model._total_classes,
                'timestamp': time.time()
            }
            
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=4)
                
            logging.info(f"✓ Checkpoint saved for task {task_id} at {checkpoint_path}")
            return True
            
        except Exception as e:
            logging.error(f"✗ Error saving checkpoint: {e}")
            return False
    
    def load_checkpoint(self, model, task_id=None, device='cuda'):
        """بارگذاری checkpoint"""
        if task_id is None:
            task_id = self.find_latest_checkpoint()
            if task_id is None:
                return None
        
        checkpoint_path = self.checkpoint_dir / f"task_{task_id}.pth"
        
        if not checkpoint_path.exists():
            logging.warning(f"Checkpoint file {checkpoint_path} does not exist")
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # بارگذاری state مدل
            model._network.load_state_dict(checkpoint['model_state_dict'])
            
            # بارگذاری stateهای دیگر
            model._known_classes = checkpoint.get('known_classes', 0)
            model._total_classes = checkpoint.get('total_classes', 0)
            model._cur_task = checkpoint.get('cur_task', 0)
            
            if 'global_prototypes' in checkpoint and checkpoint['global_prototypes'] is not None:
                model.global_prototypes = checkpoint['global_prototypes'].to(device)
            
            if 'prototype_memory' in checkpoint:
                model.prototype_memory = checkpoint['prototype_memory']
            
            # بارگذاری optimizer و scheduler
            if hasattr(model, 'optimizer') and 'optimizer_state_dict' in checkpoint:
                model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if hasattr(model, 'scheduler') and 'scheduler_state_dict' in checkpoint:
                model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # بارگذاری حافظه replay
            if hasattr(model, '_memory') and 'memory_data' in checkpoint:
                model._memory = checkpoint['memory_data']
            
            logging.info(f"✓ Checkpoint loaded from task {task_id}")
            return checkpoint
            
        except Exception as e:
            logging.error(f"✗ Error loading checkpoint: {e}")
            return None
    
    def find_latest_checkpoint(self):
        """پیدا کردن آخرین checkpoint"""
        if not self.checkpoint_dir.exists():
            return None
        
        checkpoints = [f for f in self.checkpoint_dir.iterdir() 
                      if f.name.startswith('task_') and f.name.endswith('.pth')]
        
        if not checkpoints:
            return None
        
        # استخراج شماره تسک
        task_numbers = []
        for checkpoint in checkpoints:
            try:
                task_num = int(checkpoint.stem.split('_')[1])
                task_numbers.append(task_num)
            except (ValueError, IndexError):
                continue
        
        return max(task_numbers) if task_numbers else None