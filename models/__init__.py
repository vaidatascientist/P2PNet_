from .p2pnet import build
# from .sasnet import build

# build the P2PNet model
# set training to 'True' during training
def build_model(args, training=False):
    return build(args, training)