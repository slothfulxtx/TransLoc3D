_base_ = [
    './base_cfg.py',
    './model_cfgs/transloc3d_cfg.py',
    './dataset_cfgs/university_cfg.py'
]

task_type = 'default_me'


optimizer_type = 'Adam'
optimizer_cfg = dict(
    lr=2e-5,
    weight_decay=0,
    betas=(0.9, 0.999),
)

scheduler_type = 'MultiStepLR'
scheduler_cfg = dict(
    gamma=0.2,
    milestones=(30, )
)

end_epoch = 50

train_cfg = dict(
    save_per_epoch=2,
    val_per_epoch=2,
    batch_sampler_type='ExpansionBatchSampler',
    batch_sampler_cfg=dict(
        max_batch_size=256,
        batch_size_expansion_rate=1.4,
        batch_expansion_threshold=0.7,
        batch_size=32,
        shuffle=True,
        drop_last=True,
    ),
    num_workers=16,
)

eval_cfg = dict(
    batch_sampler_cfg=dict(
        batch_size=32,
        drop_last=False,
    ),
    num_workers=16,
    normalize_embeddings=False,
)
