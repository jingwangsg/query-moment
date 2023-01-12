# ============================================================================ #
# ActivityNet Caption
# ============================================================================ #

# ============================================================================ #
# * mst_detr_v2


KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq128-span_loss5-no_fpn -co model_cfg.w_iou_loss=5 \
	model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=128 model_cfg.w_span_loss=5


KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq128-span_loss5-no_fpn-top1 -co model_cfg.w_iou_loss=5 \
	model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=128 model_cfg.w_span_loss=5 model_cfg.topk=1


KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq128-span_loss5-no_fpn-top1-mask0.15-bd123 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=128 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 model_cfg.w_bd_loss=[1,2,3] \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq128-span_loss5-no_fpn-top3-mask0.15 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=128 \
	model_cfg.w_span_loss=5 model_cfg.topk=3  \
	--wandb

# bsz 32 -> 16
KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.25 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.val_interval=0.25 \
	--wandb


KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.val_interval=0.25 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz64-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=64 train.val_interval=0.25 \
	--wandb

KN_SCHEDULE_TRAIN=1 KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz64-val0.1-patience10-lr5e-5-epoch100-train_sch10 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=100 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=1 train.optimizer.lr=5e-5 train.batch_size=64 train.val_interval=0.1 \
		train.lr_scheduler.patience=10 train.lr_scheduler.factor=0.5 \
	--wandb

KN_SCHEDULE_TRAIN=1 KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/anet/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz64-val0.1-patience10-lr5e-5-epoch100-train_sch10-enc6dec6 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=100 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=1 train.optimizer.lr=5e-5 train.batch_size=64 train.val_interval=0.1 \
		model_cfg.num_layers_enc=6 model_cfg.sr_ratio_lvls=[4,2,1,1,1,1] model_cfg.use_patch_merge=[True,True,False,False,False,False] \
		model_cfg.num_layers_dec=6 model_cfg.w_bd_loss=[0.1,0.1,0.1,0.5,0.5,2] train.lr_scheduler.patience=10 train.lr_scheduler.factor=0.5 \
		model_cfg.dropout=0.3 \
	--wandb

KN_SCHEDULE_TRAIN=1 KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/anet/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq300-span_loss5-no_fpn-top3-mask0.15-bsz16-val0.1-patience10-lr5e-5-epoch100-train_sch10-enc6dec6 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=100 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=3 train.optimizer.lr=5e-5 train.batch_size=16 train.val_interval=0.1 \
		model_cfg.num_layers_enc=6 model_cfg.sr_ratio_lvls=[4,2,1,1,1,1] model_cfg.use_patch_merge=[True,True,False,False,False,False] \
		model_cfg.num_layers_dec=6 model_cfg.w_bd_loss=[0.1,0.1,0.1,0.5,0.5,2] train.lr_scheduler.patience=10 train.lr_scheduler.factor=0.5 \
		model_cfg.dropout=0.3 \
	--wandb

KN_GROUP_LR=1 KN_SCHEDULE_TRAIN=1 KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/anet/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq300-span_loss5-no_fpn-top3-mask0.15-bsz16-val0.1-patience10-lr5e-5-epoch100-train_sch10-enc6dec6-group_lr_small \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=100 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=3 train.optimizer.lr=5e-5 train.batch_size=16 train.val_interval=0.1 \
		model_cfg.num_layers_enc=6 model_cfg.sr_ratio_lvls=[4,2,1,1,1,1] model_cfg.use_patch_merge=[True,True,False,False,False,False] \
		model_cfg.num_layers_dec=6 model_cfg.w_bd_loss=[0.1,0.1,0.1,0.5,0.5,2] train.lr_scheduler.patience=10 train.lr_scheduler.factor=0.5 \
		model_cfg.dropout=0.3 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz128-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=128 train.val_interval=0.25 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1  \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.val_interval=0.25 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-64-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.val_interval=0.25 data.max_len_video=128 \
	--wandb



KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-32-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.val_interval=0.25 data.max_len_video=64 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/anet/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz64-val0.1-lr3e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=1 train.optimizer.lr=3e-5 train.batch_size=64 train.val_interval=0.1  \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/anet/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz64-val0.1-bd111-lr1e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=1 train.optimizer.lr=1e-5 train.batch_size=64 \
		train.val_interval=0.1  model_cfg.w_bd_loss=[1,1,1] \
	--wandb

# ============================================================================ #
# Charades
# ============================================================================ #


KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz64-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=64 train.val_interval=0.25 \
	--wandb

# 测试参数敏感度
# - batch size 64 -> 32, 16
# - pooler resolution 4 -> 8, 16, 32
# - max_len_video 512 -> 1024, 256, 128
# - num query 300 -> 100 
# - num head 16 -> 32
# - w_bd_loss: [0.1, 0.5, 5] -> [0.1, 0.1, 0.3]

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.val_interval=0.25 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.25 \
	--wandb

KN_SCHEDULE_TRAIN=1 KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-train_sch10-epoch100 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=100 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.1 \
		train.lr_scheduler.patience=10 train.lr_scheduler.factor=0.5 \
	--wandb

KN_SCHEDULE_TRAIN=1 KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-train_sch10-epoch100-dec6enc6 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=100 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.1 \
		model_cfg.num_layers_enc=6 model_cfg.sr_ratio_lvls=[4,2,1,1,1,1] model_cfg.use_patch_merge=[True,True,False,False,False,False] \
		model_cfg.num_layers_dec=6 model_cfg.w_bd_loss=[0.1,0.1,0.1,0.5,0.5,2] train.lr_scheduler.patience=10 train.lr_scheduler.factor=0.5 \
		model_cfg.dropout=0.3 \
	--wandb

KN_SCHEDULE_TRAIN=1 KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr16-nq300-span_loss5-no_fpn-top3-mask0.15-bsz16-val0.1-train_sch10-epoch100-dec6enc6 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=100 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=3 train.batch_size=16 train.val_interval=0.1 \
		model_cfg.num_layers_enc=6 model_cfg.sr_ratio_lvls=[4,2,1,1,1,1] model_cfg.use_patch_merge=[True,True,False,False,False,False] \
		model_cfg.num_layers_dec=6 model_cfg.w_bd_loss=[0.1,0.1,0.1,0.5,0.5,2] train.lr_scheduler.patience=10 train.lr_scheduler.factor=0.5 \
		model_cfg.dropout=0.3 \
	--wandb

KN_GROUP_LR=1 KN_SCHEDULE_TRAIN=1 KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades16-soft_iou5-pr4-nq300-span_loss5-no_fpn-top3-mask0.15-bsz16-val0.1-train_sch10-epoch100-dec6enc6 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=100 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.1 \
		model_cfg.num_layers_enc=6 model_cfg.sr_ratio_lvls=[4,2,1,1,1,1] model_cfg.use_patch_merge=[True,True,False,False,False,False] \
		model_cfg.num_layers_dec=6 model_cfg.w_bd_loss=[0.1,0.1,0.1,0.5,0.5,2] train.lr_scheduler.patience=10 train.lr_scheduler.factor=0.5 \
		model_cfg.dropout=0.3 data.max_len_video=64 \
	--wandb




KN_GROUP_LR=1 KN_SCHEDULE_TRAIN=1 KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-train_sch10-epoch100-dec6enc6-group_lr \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=100 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.1 \
		model_cfg.num_layers_enc=6 model_cfg.sr_ratio_lvls=[4,2,1,1,1,1] model_cfg.use_patch_merge=[True,True,False,False,False,False] \
		model_cfg.num_layers_dec=6 model_cfg.w_bd_loss=[0.1,0.1,0.1,0.5,0.5,2] train.lr_scheduler.patience=10 train.lr_scheduler.factor=0.5 \
		model_cfg.dropout=0.3 \
	--wandb

KN_GROUP_LR=1 KN_SCHEDULE_TRAIN=1 KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr16-nq300-span_loss5-no_fpn-top10-mask0.15-bsz16-val0.1-train_sch10-epoch100-dec6enc6-group_lr_small \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=100 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=10 train.batch_size=16 train.val_interval=0.1 \
		model_cfg.num_layers_enc=6 model_cfg.sr_ratio_lvls=[4,2,1,1,1,1] model_cfg.use_patch_merge=[True,True,False,False,False,False] \
		model_cfg.num_layers_dec=6 model_cfg.w_bd_loss=[0.1,0.1,0.1,0.5,0.5,2] train.lr_scheduler.patience=10 train.lr_scheduler.factor=0.5 \
		model_cfg.dropout=0.3 \
	--wandb


KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr8-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=8 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.val_interval=0.25 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.val_interval=0.25 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.1-lr3e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.val_interval=0.1 train.optimizer.lr=3e-5 \
	--wandb

# bsz 64 to 32 max_len_video 512 to 1024
KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-256-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.val_interval=0.25 \
        data.max_len_video=1024 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-64-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz64-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=64 train.val_interval=0.25 \
        data.max_len_video=256 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-32-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz64-val0.1 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=64 train.val_interval=0.1 \
        data.max_len_video=128 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-32-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz64-val0.1-lr3e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=64 train.val_interval=0.1 train.optimizer.lr=3e-5 \
        data.max_len_video=128 \
	--wandb

KN_SCHEDULE_TRAIN=1 KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-32-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz64-val0.1-lr3e-5-train_sch10_0.5-epoch100 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=64 train.val_interval=0.1 train.optimizer.lr=3e-5 \
        data.max_len_video=128 train.lr_scheduler.patience=10 train.lr_scheduler.factor=0.5 \
		train.num_epochs=100 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-32-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-lr3e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.1 train.optimizer.lr=3e-5 \
        data.max_len_video=128 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-64-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz64-val0.1-lr3e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=64 train.val_interval=0.1 train.optimizer.lr=3e-5 \
        data.max_len_video=256 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-64-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-lr3e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.1 train.optimizer.lr=3e-5 \
        data.max_len_video=256 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr32-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=32 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.val_interval=0.25 \
	--wandb

# pr 4 to 16 num query 300 to 100 
KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr16-nq100-span_loss5-no_fpn-top1-mask0.15-bsz64-val0.25 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=100 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=64 train.val_interval=0.25 \
	--wandb

# nhead 16 to 32
KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz64-val0.25-nhead32 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=64 train.val_interval=0.25 model_cfg.nhead=32 \
	--wandb

# bsz 32 -> 16
# pr 4 -> 16
# w_bd_loss -> [0.1, 0.1, 0.1]
KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-bd_loss0.1x3 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
	model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.1 model_cfg.w_bd_loss=[0.1,0.1,0.1] \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-bd_loss111-lr3e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.1 model_cfg.w_bd_loss=[1,1,1] \
		train.optimizer.lr=3e-5 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-bd_loss111-lr1e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
		model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.1 model_cfg.w_bd_loss=[1,1,1] \
		train.optimizer.lr=1e-5 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-256-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-bd111-lr3e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.1 \
        data.max_len_video=1024 model_cfg.w_bd_loss=[1,1,1] train.optimizer.lr=3e-5 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-256-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-dec2-mask0.15-bsz16-val0.1-bd111-lr3e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.1 \
        data.max_len_video=1024 model_cfg.w_bd_loss=[1,1,1] train.optimizer.lr=3e-5 model_cfg.num_layers_dec=2 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-256-soft_iou5-pr4-nq300-span_loss5-no_fpn-d256-top1-mask0.15-bsz16-val0.1-bd111-lr3e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.1 \
        data.max_len_video=1024 model_cfg.w_bd_loss=[1,1,1] train.optimizer.lr=3e-5 model_cfg.d_model=256 \
	--wandb




# ============================================================================ #
# TaCOS
# ============================================================================ #
KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-tacos-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 \
	--wandb


KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-tacos-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.1 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-tacos-soft_iou5-pr4-nq100-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.1-lr3e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=100 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.optimizer.lr=3e-5 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-tacos-soft_iou5-pr4-nq50-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.1-lr3e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=50 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.optimizer.lr=3e-5 \
	--wandb


KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-tacos-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.1-lr2e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.optimizer.lr=2e-5 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-256-tacos-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.1-lr2e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.optimizer.lr=2e-5 \
		data.max_len_video=1024 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-256-tacos-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-lr2e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.optimizer.lr=2e-5 \
		data.max_len_video=1024 \
	--wandb



KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-256-tacos-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-lr3e-5-epoch100-patience10 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.optimizer.lr=3e-5 \
		train.num_epochs=100 train.lr_scheduler.patience=10 \
		data.max_len_video=1024 \
	--wandb



KN_SCHEDULE_TRAIN=1 KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-256-tacos-soft_iou5-pr16-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-lr3e-5-epoch100-train_sch10 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.optimizer.lr=3e-5 \
		train.num_epochs=100 train.lr_scheduler.patience=10 \
		data.max_len_video=1024 \
	--wandb

KN_SCHEDULE_TRAIN=1 KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-256-tacos-soft_iou5-pr16-nq300-span_loss5-no_fpn-top3-mask0.15-bsz16-val0.1-lr3e-5-epoch100-train_sch10 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=3 train.batch_size=16 train.optimizer.lr=3e-5 \
		train.num_epochs=100 train.lr_scheduler.patience=10 \
		data.max_len_video=1024 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-256-tacos-soft_iou5-pr16-nq300-d256-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-lr2e-5-enc6dec6 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.optimizer.lr=2e-5 \
		data.max_len_video=1024 model_cfg.d_model=256 \
		model_cfg.num_layers_enc=6 model_cfg.sr_ratio_lvls=[4,2,1,1,1,1] model_cfg.use_patch_merge=[True,True,False,False,False,False] \
		model_cfg.num_layers_dec=6 model_cfg.w_bd_loss=[0.1,0.1,0.1,0.5,0.5,2] train.lr_scheduler.patience=10 train.lr_scheduler.factor=0.5 \
		model_cfg.dropout=0.3 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-256-tacos-soft_iou5-pr16-nq300-dec2-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-lr2e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=16 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.optimizer.lr=2e-5 \
		data.max_len_video=1024 model_cfg.num_layers_dec=2 \
	--wandb


KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-tacos-soft_iou5-pr4-nq50-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.1-bd111-lr1e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=50 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 \
		train.optimizer.lr=1e-5 model_cfg.w_bd_loss=[1,1,1] \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/tacos/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-tacos-soft_iou5-pr4-nq100-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.1-bd111-lr1e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=100 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 \
		train.optimizer.lr=1e-5 model_cfg.w_bd_loss=[1,1,1] \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-256-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.1-lr2e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.val_interval=0.1 \
        data.max_len_video=1024 model_cfg.w_bd_loss=[1,1,1] train.optimizer.lr=2e-5 \
	--wandb

KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-256-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz32-val0.1-lr2e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=32 train.val_interval=0.1 \
        data.max_len_video=1024 model_cfg.w_bd_loss=[1,0.5,0.3] train.optimizer.lr=2e-5 \
	--wandb


KN_USE_IOU=1 KN_SPAN_LOSS=1 KN_SOFT_IOU=1 python main.py config/mst_detr_v2/charades/mst_detr_v2-no_fpn.py \
	--exp mst_detr_v2-charades-256-soft_iou5-pr4-nq300-span_loss5-no_fpn-top1-mask0.15-bsz16-val0.1-lr2e-5 \
	-co model_cfg.w_iou_loss=5 model.head.pooler_resolution=4 train.num_epochs=50 model_cfg.num_query=300 \
	    model_cfg.w_span_loss=5 model_cfg.topk=1 train.batch_size=16 train.val_interval=0.1 \
        data.max_len_video=1024 model_cfg.w_bd_loss=[1,1,1] train.optimizer.lr=2e-5 \
	--wandb

# ============================================================================ #
# * mst_detr_v4
# ============================================================================ #

# ============================================================================ #
# anet

python main.py config/mst_detr_v4/anet/mst_detr_v4.py --exp mst_detr_v4-anet --wandb

python main.py config/mst_detr_v4/anet/mst_detr_v4.py --exp mst_detr_v4-anet-bsz64 \
	-co train.batch_size=64 \
	--wandb

KN_GROUP_LR=[1,0.5,0.3] KN_SCHEDULE_TRAIN=1 \
python main.py config/mst_detr_v4/anet/mst_detr_v4.py --exp mst_detr_v4-anet-bsz64-train_sch10-group_lr \
	-co train.batch_size=64 train.optimizer.lr=5e-5 \
	--wandb

# ============================================================================ #
# charades

python main.py config/mst_detr_v4/charades/mst_detr_v4.py --exp mst_detr_v4-charades --wandb

python main.py config/mst_detr_v4/charades/mst_detr_v4.py --exp mst_detr_v4-charades-bsz64 \
	-co train.batch_size=64 \
	--wandb

KN_GROUP_LR=1 KN_SCHEDULE_TRAIN=1 \
python main.py config/mst_detr_v4/charades/mst_detr_v4.py --exp mst_detr_v4-charades-bsz16-pr16-val0.1-train_sch10-group_lr_small \
	-co train.batch_size=16 train.val_interval=0.1 \
		model_cfg.num_layers_enc=6 model_cfg.sr_ratio_lvls=[4,2,1,1,1,1] model_cfg.use_patch_merge=[True,True,False,False,False,False] \
		model_cfg.num_layers_dec=6 model_cfg.w_bd_loss=[0.1,0.1,0.1,0.5,0.5,2] train.lr_scheduler.patience=10 train.lr_scheduler.factor=0.5 \
		model_cfg.dropout=0.3 model.head.pooler_resolution=16 \
	--wandb

KN_GROUP_LR=[1,0.5,0.3] KN_SCHEDULE_TRAIN=1 \
python main.py config/mst_detr_v4/charades/mst_detr_v4.py --exp mst_detr_v4-charades-cross-train_sch10-group_lr \
	-co train.optimizer.lr=5e-5
	--wandb

KN_GROUP_LR=[1,0.5,0.3] KN_SCHEDULE_TRAIN=1 \
python main.py config/mst_detr_v4/charades/mst_detr_v4.py --exp mst_detr_v4-charades-cross-pr16-train_sch10-group_lr \
	-co train.optimizer.lr=5e-5 model.head.pooler_resolution=16 \
	--wandb

KN_GROUP_LR=[1,0.5,0.5,0.3,0.3] KN_SCHEDULE_TRAIN=1 \
python main.py config/mst_detr_v4/charades/mst_detr_v4.py --exp mst_detr_v4-charades-cross-pr16-train_sch10-group_lr-dec5 \
	-co train.optimizer.lr=5e-5 model_cfg.num_layers_dec=5  model.head.pooler_resolution=16 \
	--wandb

# ============================================================================ #
# tacos


