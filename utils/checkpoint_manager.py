import os
import torch
import logging
import json

class CheckpointManager:
    def __init__(self, checkpoint_dir="./checkpoints", auto_save=True):
        self.checkpoint_dir = checkpoint_dir
        self.auto_save = auto_save
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, model, task_id, metrics=None):
        """ذخیره کامل وضعیت مدل پس از هر تسک"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"task_{task_id}.pth")
        
        checkpoint_data = {
            'model_state_dict': model._network.state_dict(),
            'task_id': task_id,
            'known_classes': model._known_classes,
            'global_prototypes': model.global_prototypes.cpu() if hasattr(model, 'global_prototypes') else None,
            'prototype_memory': model.prototype_memory if hasattr(model, 'prototype_memory') else {},
            'metrics': metrics or {}
        }
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            
            # همچنین اطلاعات متا را ذخیره کنید
            meta_path = os.path.join(self.checkpoint_dir, "training_meta.json")
            meta_data = {
                'last_task': task_id,
                'total_classes': model._total_classes if hasattr(model, '_total_classes') else 0,
                'timestamp': torch.tensor(torch.timestamp())
            }
            
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f)
                
            logging.info(f"✓ Checkpoint saved for task {task_id}")
            return True
            
        except Exception as e:
            logging.error(f"✗ Error saving checkpoint: {e}")
            return False
    
    def load_checkpoint(self, model, task_id=None, device='cuda'):
        """بارگذاری checkpoint"""
        if task_id is None:
            task_id = self.find_latest_checkpoint()
            if task_id is None:
                return False
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"task_{task_id}.pth")
        
        if not os.path.exists(checkpoint_path):
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            model._network.load_state_dict(checkpoint['model_state_dict'])
            model._known_classes = checkpoint.get('known_classes', 0)
            
            if 'global_prototypes' in checkpoint and checkpoint['global_prototypes'] is not None:
                model.global_prototypes = checkpoint['global_prototypes'].to(device)
            
            if 'prototype_memory' in checkpoint:
                model.prototype_memory = checkpoint['prototype_memory']
            
            logging.info(f"✓ Checkpoint loaded from task {task_id}")
            return task_id
            
        except Exception as e:
            logging.error(f"✗ Error loading checkpoint: {e}")
            return False
    
    def find_latest_checkpoint(self):
        """پیدا کردن آخرین checkpoint"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith('task_') and f.endswith('.pth')]
        
        if not checkpoints:
            return None
        
        # استخراج شماره تسک
        task_numbers = []
        for checkpoint in checkpoints:
            try:
                task_num = int(checkpoint.split('_')[1].split('.')[0])
                task_numbers.append(task_num)
            except:
                continue
        
        return max(task_numbers) if task_numbers else None
    
    def get_checkpoint_info(self, task_id):
        """دریافت اطلاعات checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"task_{task_id}.pth")
        if os.path.exists(checkpoint_path):
            return {
                'path': checkpoint_path,
                'size': os.path.getsize(checkpoint_path),
                'modified': os.path.getmtime(checkpoint_path)
            }
        return None