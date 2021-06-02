from .default_me_task import DefaultMETask


def create_task(task_type, cfg, log):
    type2task = dict(
        default_me=DefaultMETask,
    )
    return type2task[task_type](cfg, log)
