from tensorboardX import SummaryWriter
from nets.yolo4_tiny import YoloBody
from utils.config import cfg_mnet, cfg_re50
import torch

cfg = cfg_re50
net = YoloBody(cfg=cfg, mode='eval',phi=3).eval()

dummy_input = torch.rand(1, 3, 608, 608)
writer = SummaryWriter(comment='yolov4tiny-face')
writer.add_graph(model=net,input_to_model=(dummy_input,))


