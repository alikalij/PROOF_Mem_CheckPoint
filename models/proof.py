import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import Proof_Net
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss
import os
import gc


num_workers = 0

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        self._train_transformer = False
        self._network = Proof_Net(args, False)

        # تنظیمات بهینه‌سازی عملکرد
        self.feat_dim = 512  # بعد ثابت برای ویژگی‌های CLIP

        # پارامترهای آموزش
        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.frozen_layers = get_attribute(args, "frozen_layers", None)
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)
        
        self._known_classes = 0
        self.use_cos = get_attribute(args, "use_cos", False)
        
        # مدیریت حافظه پیشرفته
        self.global_prototypes = None
        self.prototype_memory = {}
        self.prototype_update_factor = 0.7

        # مسیرهای checkpoint - باید از خارج تنظیم شوند
        self.checkpoint_dir = None

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
    
    def cal_prototype(self, trainloader, model):
        """محاسبه پروتوتایپ با مدیریت حافظه بهینه"""
        model.eval()
        prototype_dict = {}
        
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                # مدیریت حافظه
                if i % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                (_, data, label) = batch
                data = data.to(self._device)
                embedding = model.convnet.encode_image(data, True)
                
                # انتقال به CPU برای صرفه‌جویی در حافظه GPU
                embedding = embedding.cpu()
                label = label.cpu()
                
                for cls in label.unique():
                    cls = cls.item()
                    mask = (label == cls)
                    cls_embeddings = embedding[mask]
                    
                    if cls not in prototype_dict:
                        prototype_dict[cls] = []
                    prototype_dict[cls].append(cls_embeddings.mean(0))
                
                # آزاد کردن حافظه
                del data, embedding, label
                
        # محاسبه میانگین نهایی
        for cls, embeddings_list in prototype_dict.items():
            final_proto = torch.stack(embeddings_list).mean(0)
            self._network.img_prototypes[cls] = final_proto.to(self._device)
            self.prototype_memory[cls] = final_proto

    def incremental_train(self, data_manager):
        self._cur_task += 1        
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info(f"log3===========")
        # به‌روزرسانی پروتوتایپ‌ها در شبکه
        self._network.update_prototype(self._total_classes)
        logging.info(f"log4===========")
        # مقداردهی اولیه global_prototypes
        if self.global_prototypes is None:
            self.global_prototypes = torch.zeros(
                (self._total_classes, self.feat_dim),
                device=self._device
            )
            logging.info(f"log5===========")
        else:
            current_size = self.global_prototypes.shape[0]
            if current_size < self._total_classes:
                new_prototypes = torch.zeros(
                    (self._total_classes - current_size, self.feat_dim),
                    device=self._device
                )
                self.global_prototypes = torch.cat([self.global_prototypes, new_prototypes], dim=0)
                logging.info(f"log6===========")
        self._network.update_context_prompt()
        logging.info(f"log7===========")
        self._network.extend_task()
        
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train", appendent=self._get_memory()
        )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self._network.to(self._device)
        logging.info(f"log8===========")
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )
        logging.info(f"log9===========")
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        logging.info(f"log10===========")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
        )
        logging.info(f"log11===========")
        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test"
        )
        logging.info(f"log12===========")
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )
        logging.info(f"log13===========")
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        logging.info(f"log14===========")
        self.cal_prototype(self.train_loader_for_protonet, self._network)
        logging.info(f"log15===========")
        # Update global prototypes with new knowledge
        for class_idx in range(self._known_classes, self._total_classes):
            if class_idx < self._network.img_prototypes.shape[0]:
                self.global_prototypes[class_idx] = self._network.img_prototypes[class_idx].clone()
        logging.info(f"log16===========")
        self._train_proj(self.train_loader, self.test_loader)
        logging.info(f"log17===========")
        # پاکسازی حافظه بعد از هر تسک
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        logging.info(f"log18===========")
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train_proj(self, train_loader, test_loader):
        self._train_transformer = True
        self._network.to(self._device)
        logging.info(f"log21===========")
        # Freeze appropriate layers
        for name, param in self._network.convnet.named_parameters():
            if 'logit_scale' not in name:
                param.requires_grad = False
        self._network.freeze_projection_weight_new()
        logging.info(f"log22===========")
        # Optimizer setup
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, 
                                 lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam': 
            optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, 
                                   weight_decay=self.weight_decay)
        logging.info(f"log23===========")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr
        )
        logging.info(f"log24===========")
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]
        prog_bar = tqdm(range(self.tuned_epoch))
        cliploss = ClipLoss()
        logging.info(f"log25===========")
        total_labels = class_to_label[:self._total_classes]
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            
            # نظارت بر حافظه در هر epoch
            self.monitor_memory()
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                # پاکسازی حافظه هر 10 batch
                if i % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                labels = [class_to_label[y] for y in targets]
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                
                # Text features
                texts = [templates.format(inst) for inst in total_labels]
                texts = self._network.tokenizer(texts).to(self._device)
                text_features = self._network.encode_text(texts)
                text_feas = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Image features
                image_features = self._network.encode_image(inputs)
                img_feas = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Forward pass
                image_features, text_features, logit_scale, _ = self._network.forward_transformer(
                    img_feas, text_feas, self._train_transformer
                )
                logits = image_features @ text_features.T
                
                # Enhanced prototype loss using global memory
                proto_outputs = image_features @ self.global_prototypes.T
                protoloss = F.cross_entropy(proto_outputs, targets)
                
                # CLIP consistency loss
                clip_texts = [templates.format(inst) for inst in labels]
                clip_text_feas = self._network.encode_text(self._network.tokenizer(clip_texts).to(self._device))
                clip_text_feas = clip_text_feas / clip_text_feas.norm(dim=-1, keepdim=True)
                clip_loss = cliploss(img_feas, clip_text_feas, logit_scale)
                
                # Classification loss
                loss = F.cross_entropy(logits, targets)
                
                # Combined loss with enhanced weights
                #total_loss = loss + 0.5 * clip_loss + 0.7 * protoloss
                total_loss = loss + 0.4 * clip_loss + 0.6 * protoloss
                #task_factor = self._cur_task / self.args["num_tasks"]
                #total_loss = loss + (0.6 - 0.2 * task_factor) * clip_loss + (0.4 + 0.2 * task_factor) * protoloss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                losses += total_loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                self._cur_task, epoch + 1, self.args['tuned_epoch'], 
                losses / len(train_loader), train_acc, test_acc
            )
            prog_bar.set_description(info)
        logging.info(f"log29===========")
    def _compute_accuracy(self, model, loader):
        model.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]
        
        # Prepare text features
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = model.tokenizer(texts).to(self._device)
                class_embeddings = model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
        
        # Evaluation with enhanced prototypes
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # Extract features
                image_features = model.encode_image(inputs)
                transf_image_features, transf_text_features, _, _ = model.forward_transformer(
                    image_features, text_features, self._train_transformer
                )
                
                # Enhanced prediction with global prototypes
                original_outputs = image_features @ text_features.T
                transformer_outputs = transf_image_features @ transf_text_features.T
                proto_outputs = transf_image_features @ self.global_prototypes.T
                
                outputs = original_outputs + transformer_outputs + proto_outputs

            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    def _eval_cnn(self, loader):
        """
        متد ارزیابی سفارشی برای مدل PROOF که همزمان از ویژگی‌های تصویری و متنی استفاده می‌کند
        """
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]
        
        # آماده‌سازی ویژگی‌های متنی
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # استخراج ویژگی‌های تصویری
                image_features = self._network.encode_image(inputs)
                
                # پردازش ترنسفورمری
                transf_image_features, transf_text_features, _, proto_feas = self._network.forward_transformer(
                    image_features, text_features, False
                )
                
                # محاسبه خروجی‌ها
                original_outputs = image_features @ text_features.T
                transformer_outputs = transf_image_features @ transf_text_features.T
                proto_outputs = transf_image_features @ proto_feas.T
                
                # ترکیب خروجی‌ها
                outputs = original_outputs + transformer_outputs + proto_outputs

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
    
    def monitor_memory(self):
        """نظارت بر مصرف حافظه و پاکسازی دوره‌ای"""
        try:
            import psutil
            memory_usage = psutil.virtual_memory().percent
            
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            else:
                gpu_usage = 0
            
            if memory_usage > 85 or gpu_usage > 85:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Memory cleaned - RAM: {memory_usage}%, GPU: {gpu_usage}%")
                
        except ImportError:
            # اگر psutil نصب نبود، فقط حافظه GPU را پاکسازی کن
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()