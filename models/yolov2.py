import torch
from torch import nn
import torchvision.models as tm
from torch.autograd import Variable
from models.nn import DetectNet
from models.layers import conv2d, max_pool, ConvBnAct, SpaceToDepth
import os
import numpy as np
from collections import OrderedDict


class YOLO(DetectNet):

    def __init__(self, input_shape, num_classes, anchors, **kwargs):

        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.d = self._prepare_module()
        self.is_cuda = torch.cuda.is_available()

        grid_size = [x // 32 for x in input_shape[1:]]
        grid_h, grid_w = grid_size
        grid_wh = torch.tensor([grid_w, grid_h]).view(1, 1, 2, 1, 1).float()
        cxcy = np.transpose([np.tile(np.arange(grid_w), grid_h),
                             np.repeat(np.arange(grid_h), grid_h)])
        cxcy = np.transpose(cxcy, [1, 0])
        cxcy = np.reshape(cxcy, (1, 1, 2, grid_h, grid_w))
        cxcy = torch.tensor(cxcy).float()

        anchors = torch.tensor(anchors).float()
        pwph = anchors.view(1, len(anchors), 2, 1, 1) / 32

        self.cxcy = Variable(cxcy, requires_grad=False)
        self.pwph = Variable(pwph, requires_grad=False)
        self.grid_wh = Variable(grid_wh, requires_grad=False)
        if self.is_cuda:
            self.cuda()
            self.cxcy = self.cxcy.cuda()
            self.pwph = self.pwph.cuda()
            self.grid_wh = self.grid_wh.cuda()

        self.apply(self._init_normal)
    def _prepare_module(self):

        d = OrderedDict()

        #conv1 - batch_norm1 - leaky_relu1 - pool1
        d['conv1'] = ConvBnAct(3, 32, 3, stride=1, padding=1)
        d['pool1'] = max_pool(2, 2)

        #conv2 - batch_norm2 - leaky_relu2 - pool2
        d['conv2'] = ConvBnAct(32, 64, 3, stride=1, padding=1)
        d['pool2'] = max_pool(2, 2)

        #conv3 - batch_norm3 - leaky_relu3
        d['conv3'] = ConvBnAct(64, 128, 3, stride=1, padding=1)

        #conv4 - batch_norm4 - leaky_relu4
        d['conv4'] = ConvBnAct(128, 64, 1, stride=1, padding=0)

        #conv5 - batch_norm5 - leaky_relu5 - pool5
        d['conv5'] = ConvBnAct(64, 128, 3, stride=1, padding=1)
        d['pool5'] = max_pool(2, 2)

        #conv6 - batch_norm6 - leaky_relu6
        d['conv6'] = ConvBnAct(128, 256, 3, stride=1, padding=1)

        #conv7 - batch_norm7 - leaky_relu7
        d['conv7'] = ConvBnAct(256, 128, 1, stride=1, padding=0)

        #conv8 - batch_norm8 - leaky_relu8 - pool8
        d['conv8'] = ConvBnAct(128, 256, 3, stride=1, padding=1)
        d['pool8'] = max_pool(2, 2)

        #conv9 - batch_norm9 - leaky_relu9
        d['conv9'] = ConvBnAct(256, 512, 3, stride=1, padding=1)

        #conv10 - batch_norm10 - leaky_relu10
        d['conv10'] = ConvBnAct(512, 256, 1, stride=1, padding=0)

        #conv11 - batch_norm11 - leaky_relu11
        d['conv11'] = ConvBnAct(256, 512, 3, stride=1, padding=1)

        #conv12 - batch_norm12 - leaky_relu12
        d['conv12'] = ConvBnAct(512, 256, 1, stride=1, padding=0)

        #conv13 - batch_norm13 - leaky_relu13 - pool13
        d['conv13'] = ConvBnAct(256, 512, 3, stride=1, padding=1)
        d['pool13'] = max_pool(2, 2)

        #conv14 - batch_norm14 - leaky_relu14
        d['conv14'] = ConvBnAct(512, 1024, 3, stride=1, padding=1)

        #conv15 - batch_norm15 - leaky_relu15
        d['conv15'] = ConvBnAct(1024, 512, 1, stride=1, padding=0)

        #conv16 - batch_norm16 - leaky_relu16
        d['conv16'] = ConvBnAct(512, 1024, 3, stride=1, padding=1)

        #conv17 - batch_norm16 - leaky_relu17
        d['conv17'] = ConvBnAct(1024, 512, 1, stride=1, padding=0)

        #conv18 - batch_norm18 - leaky_relu18
        d['conv18'] = ConvBnAct(512, 1024, 3, stride=1, padding=1)

        #conv19 - batch_norm19 - leaky_relu19
        d['conv19'] = ConvBnAct(1024, 1024, 3, stride=1, padding=1)

        # Detection Layer
        #conv20 - batch_norm20 - leaky_relu20
        d['conv20'] = ConvBnAct(1024, 1024, 3, stride=1, padding=1)

        # concatenate layer20 and layer 13 using space to depth
        d['skip_connection'] = nn.Sequential(
            ConvBnAct(512, 64, 1, stride=1, padding=0), SpaceToDepth(2))
        d['conv21'] = ConvBnAct(1024, 1024, 3, stride=1, padding=1)

        #conv22 - batch_norm22 - leaky_relu22
        d['conv22'] = ConvBnAct(1280, 1024, 3, stride=1, padding=1)

        output_channel = self.num_anchors * (5 + self.num_classes)
        d['logits'] = conv2d(1024, output_channel, 1,
                             stride=1, padding=0, bias=True)

        self.module = nn.ModuleList()
        for i in d.values():
            self.module.append(i)
        return d

    def forward(self, x):

        d = self.d
        out = x
        for name in d:
            if name == 'conv13':
                out = d[name](out)
                skip = out
            elif name == 'skip_connection':
                skip = d[name](skip)
            elif name == 'conv21':
                out = torch.cat((d[name](out), skip), dim=1)
            else:
                out = d[name](out)
        out = out.view(-1, self.num_anchors, (5 + self.num_classes),
                       out.size()[-2], out.size()[-1])

        return out

    def _decode_output(self, x):
        num_classes = self.num_classes
        cxcy = self.cxcy
        pwph = self.pwph
        pred = self.forward(x)

        txty, twth = pred[:, :,  0:2, :, :], pred[:, :, 2:4, :, :]
        confidence = torch.sigmoid(pred[:, :, 4:5, :, :])
        class_probs = F.softmax(pred[:, :, 5:, :, :], dim=2)\
            if num_classes > 1 else torch.sigmoid(pred[:, :, 5:, :, :])
        bxby = torch.sigmoid(txty) + cxcy
        bwbh = torch.exp(twth) * pwph

        return bxby, bwbh, confidence, class_probs

    def output(self, x):
        grid_wh = self.grid_wh
        bxby, bwbh, confidence, class_probs = self._decode_output(x)
        # calculating for prediction
        nxny, nwnh = bxby / grid_wh, bwbh / grid_wh
        nx1ny1, nx2ny2 = nxny - 0.5 * nwnh, nxny + 0.5 * nwnh
        pred_y = torch.cat((nx1ny1, nx2ny2, confidence, class_probs), dim=2)
        pred_y = pred_y.permute(0, 3, 4, 1, 2)
        return pred_y

    def loss(self, x, y):
        loss_weights = [5, 5, 5, 0.5, 1.0]  # FIXME
        grid_wh = self.grid_wh
        y = y.permute(0, 3, 4, 1, 2)

        bxby, bwbh, confidence, class_probs = self._decode_output(x)

        gt_bxby = 0.5 * (y[:, :, 0:2, :, :] + y[:, :, 2:4, :, :]) * grid_wh
        gt_bwbh = (y[:, :, 2:4, :, :] - y[:, :, 0:2, :, :]) * grid_wh

        resp_mask = y[:, :, 4:5, :, :]
        no_resp_mask = 1.0 - resp_mask
        gt_confidence = resp_mask
        gt_class_probs = y[:, :, 5:, :, :]
        loss_bxby = loss_weights[0] * resp_mask * \
            torch.pow((gt_bxby - bxby), 2)
        loss_bwbh = loss_weights[1] * resp_mask * \
            torch.pow((gt_bwbh.sqrt() - bwbh.sqrt()), 2)
        loss_resp_conf = loss_weights[2] * resp_mask * \
            torch.pow((gt_confidence - confidence), 2)
        loss_no_resp_conf = loss_weights[
            3] * no_resp_mask * torch.pow((gt_confidence - confidence), 2)
        loss_class_probs = loss_weights[
            4] * resp_mask * torch.pow((gt_class_probs - class_probs), 2)

        merged_loss = torch.cat((
                                loss_bxby,
                                loss_bwbh,
                                loss_resp_conf,
                                loss_no_resp_conf,
                                loss_class_probs
                                ),
                                dim=2)
        total_loss = torch.sum(merged_loss, dim=2)
        total_loss = torch.mean(merged_loss)
        return total_loss
