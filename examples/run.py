import detr
import detr.util.misc as utils
from detr.engine import evaluate, train_one_epoch
import torch
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path

#backbone = detr.build_backbone(args)
#transformer = detr.build_transformer(args)
#model = detr.DETR(num_classes=91,
#                  num_queries=num_queries,
#                  aux_loss=aux_loss)
dataset_file = "coco"
coco_path = "../detr/data"
hidden_dim = 256
position_embedding = 'sine'
lr_backbone = 1e-5
lr = 1e-4
weight_decay = 1e-4
lr_drop = 200
masks = False
backbone = "resnet50"
dilation = False
num_classes = 91
device = "cuda"
dropout = 0.1
nheads = 8
dim_feedforward = 2048
enc_layers = 6
dec_layers = 6
pre_norm = False
no_aux_loss = True
num_queries = 100
frozen_weights = None
giou_loss_coef = 2
bbox_loss_coef = 5
mask_loss_coef = 1
dice_loss_coef = 1
eos_coef = 0.1
set_cost_class = 1
set_cost_bbox = 5
set_cost_giou = 2
distributed = False
batch_size = 4
num_workers = 6
output_dir = "log"
epochs=300
clip_max_norm=0.1 

model, criterion, postprocessors = detr.build(dataset_file, hidden_dim, position_embedding,
                                              lr_backbone, masks, backbone, dilation,
                                              num_classes, num_queries, device, dropout, nheads,
                                              dim_feedforward, enc_layers, dec_layers, pre_norm, no_aux_loss,
                                              frozen_weights, giou_loss_coef, bbox_loss_coef, mask_loss_coef,
                                              dice_loss_coef, eos_coef, set_cost_class, set_cost_bbox, set_cost_giou)
model_without_ddp = model
device_ids = range(torch.cuda.device_count())
if distributed:
    model = torch.nn.parallel.DataParallel(model, device_ids=device_ids)
    model_without_ddp = model.module
model.to(device)
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

param_dicts = [
    {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": lr_backbone,
    },
]

optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                              weight_decay = weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)

dataset_train = detr.build_coco('train', coco_path, masks)
dataset_val = detr.build_coco('val', coco_path, masks)
sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, batch_size, drop_last=True)

data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                               collate_fn=utils.collate_fn, num_workers=num_workers)
data_loader_val = DataLoader(dataset_val, batch_size, sampler=sampler_val,
                             drop_last=False, collate_fn=utils.collate_fn, num_workers=num_workers)


base_ds = detr.datasets.get_coco_api_from_dataset(dataset_val)

if frozen_weights is not None:
    checkpoint = torch.load(frozen_weights, map_location='cpu')
    model_without_ddp.detr.load_state_dict(checkpoint['model'])

output_dir = Path(output_dir)

print("Start training")
for epoch in range(epochs):
    train_stats = train_one_epoch(
        model, criterion, data_loader_train, optimizer, device, epoch,
        clip_max_norm)
    lr_scheduler.step()
