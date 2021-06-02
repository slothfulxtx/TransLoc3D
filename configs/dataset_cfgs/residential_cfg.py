dataset_type = 'Residential'
data_root_dir = 'data/benchmark_datasets/'
dataset_cfg = dict(
    data_root_dir='data/benchmark_datasets/',
    transform_cfg=dict(
        train=[
            dict(
                type='Jitter3D',
                objs=['pcd'],
                params=dict(
                    sigma=0.001,
                    clip=0.002,
                ),
            ),
            dict(
                type='Drop3D',
                objs=['pcd'],
                params=dict(
                    method='random',
                    min_dr=0.0,
                    max_dr=0.1,
                ),
            ),
            dict(
                type='Translation3D',
                objs=['pcd'],
                params=dict(
                    max_delta=0.01,
                ),
            ),
            dict(
                type='Drop3D',
                objs=['pcd'],
                params=dict(
                    method='cuboid',
                    p=0.4,
                    bbox_pc_name='pcd'
                ),
            ),
            dict(
                type='ToTensor',
                objs=['pcd'],
                params=dict(),
            )
        ],
        database=[
            dict(
                type='ToTensor',
                objs=['pcd'],
                params=dict(),
            )
        ],
        queries=[
            dict(
                type='ToTensor',
                objs=['pcd'],
                params=dict(),
            )
        ],
    ),
    batch_transform_cfg=dict(
        train=[
            dict(
                type='Rotation3D',
                objs=['pcd'],
                params=dict(
                    batch=True,
                    max_theta1=5,
                    max_theta2=0,
                    axis=[0, 0, 1]
                ),
            ),
            dict(
                type='Mirror3D',
                objs=['pcd'],
                params=dict(
                    batch=True,
                    method='xyz_plane',
                    p=[0.25, 0.25, 0.]
                ),
            ),
        ],
        database=[
        ],
        queries=[
        ],
    ),
    train_catalog_file_path=data_root_dir+'training_queries_refine2.pickle',
    cached_train_catalog_file_path=data_root_dir +
    'training_queries_refine2_cached.pickle',
    database_file_path=data_root_dir+'residential_evaluation_database.pickle',
    queries_file_path=data_root_dir+'residential_evaluation_query.pickle',
)
