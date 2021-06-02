model_type = 'TransLoc3D'
model_cfg = dict(
    backbone_cfg=dict(
        up_conv_cfgs=[
            dict(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=1,
            ),
            dict(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=2,
            ),
        ],
        transformer_cfg=dict(
            num_attn_layers=6,
            global_channels=64,
            local_channels=0,
            num_centers=[256, 128, 128, 64, 64, 64],
            num_heads=2
        ),
        in_channels=1,
        out_channels=512
    ),
    pool_cfg=dict(
        type='NetVlad',
        in_channels=512,
        out_channels=256,
        cluster_size=64,
        gating=True,
        add_bn=True
    ),
    quantization_size=0.01,
)
