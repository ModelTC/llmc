import gc

import torch
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class AccuracyEval:
    def __init__(self, config):
        self.eval_config = config.eval
        self.imagenet_root = self.eval_config['path']
        self.bs = self.eval_config['bs']
        self.num_workers = self.eval_config.get('num_workers', 8)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_imagenet(self):
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        val_dataset = ImageFolder(root=self.imagenet_root, transform=val_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: x,
            pin_memory=True
        )
        return val_loader

    def eval(self, model):
        self.model = model.get_model()
        self.processor = model.processor
        self.model.eval()
        self.model.to(self.device)

        val_loader = self.load_imagenet()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                logger.info(f'Processed batch {batch_idx + 1}/{len(val_loader)}')
                imgs, labels = zip(*batch)
                labels = torch.tensor(labels).to(self.device)
                inputs = self.processor(images=list(imgs), return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

        self.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        return accuracy
