# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dCoST',
        pretrained2d=True,
        pretrained='torchvision://resnet101',
        depth=101,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=2,
        pool1_stride_t=2,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((0, 1, 0), (1, 0, 1, 0),
                 (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),  # noqa
                 (0, 1, 0)),
        inflate_style='costb',
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))