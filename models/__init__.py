from .transloc3d.model import TransLoc3D


def create_model(model_type, model_cfg):
    type2model = dict(
        TransLoc3D=TransLoc3D,
    )
    return type2model[model_type](model_cfg)
