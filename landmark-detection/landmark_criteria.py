import torch
import torch.nn as nn
from landmark_detection import HRNet_lms


class DiffPos(nn.Module):
    def __init__(self, device='cuda:0'):
        super(DiffPos, self).__init__()
        self.s = 64
        
        x = torch.arange(0,self.s,1)
        y = torch.arange(0,self.s,1)

        yy, xx  = torch.meshgrid(x, y)
        xx = xx.view(-1,1,64,64).to('cuda:0')
        yy = yy.view(-1,1,64,64).to('cuda:0')
        
        # setting the x y coorinate
        self.register_buffer('yy', yy)
        self.register_buffer('xx', xx)

        self.thresh = nn.Threshold(0.1, 0)
    
    def forward(self, heatmaps):
        # binarize the heatmap
        heatmaps = self.thresh(heatmaps)

        # get center of the activated point
        x_pos = torch.sum(heatmaps * self.xx, dim=[2,3]) / torch.sum(heatmaps, dim=[2,3])
        y_pos = torch.sum(heatmaps * self.yy, dim=[2,3]) / torch.sum(heatmaps, dim=[2,3])
        return x_pos/self.s, y_pos/self.s
    

class LandmarkCriterion():
    def __init__(self, device, multi_gpu=True):
        # super(LandmarkCriterion, self).__init__()
        self.net = HRNet_lms().to(device)
        self.net.eval()
        if multi_gpu:
            self.net = torch.nn.DataParallel(self.net)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.pose_extract = DiffPos(device=device)

        # left  eye lms index : 36~41
        # right eye lms index : 42~47
        # mouth eye lms index : 48~67


    def get_heatmap(self, x, no_grad=False):        
        # get heatmap
        if no_grad:
            with torch.no_grad():
                heat = self.net(x)
        else:
            heat = self.net(x)
        return heat

    def lms_loss(self, heat, heat_hat, is_eye=False, is_mouth_middle=False, is_mouth_out=False, is_mouth=False,all_lms=False):
        # compute heatmap loss
        idx = []
        if all_lms:
            idx += [i for i in range(0,68)]
        else:
            if is_eye:
                idx += [i for i in range(36,48)]
            if is_mouth_middle:
                idx += [51,57,62,66]
            if is_mouth_out:
                idx += [48,54]
            if is_mouth:
                idx += [i for i in range(48,68)]
            if not is_eye and not is_mouth_middle and not is_mouth_out and not is_mouth:

                return torch.tensor(0.).to('cuda:0')
       
        heat_     = heat[:,idx]
        heat_hat_ = heat_hat[:,idx]
        
        heatmap_loss = self.criterion(heat_, heat_hat_)
        return heatmap_loss

    def lms_coordinate_loss(self, heat, heat_hat, \
        only_eye=False, only_mouth=False\
        ):
        bs = heat.size(0)
        # get position of lms
        if only_eye:
            heat_     = heat[:,36:48]
            heat_hat_ = heat_hat[:,36:48]
        elif only_mouth:
            mouth     = [51,57,62,66] # middle mouth lms
            heat_     = heat[:,mouth]
            heat_hat_ = heat_hat[:,mouth]
            # heat_     = heat[:,48:68]
            # heat_hat_ = heat_hat[:,48:68]
        else:
            heat_     = heat
            heat_hat_ = heat_hat
        x_pos    , y_pos     = self.pose_extract(heat_)
        x_pos_hat, y_pos_hat = self.pose_extract(heat_hat_)

        ### pose loss  -----------------------------------------------------------
        pos_loss  = self.criterion(x_pos, x_pos_hat)
        pos_loss += self.criterion(y_pos, y_pos_hat)

        ### first order loss  ---------------------------------------------------
        # compute the difference in time
        x_vel = x_pos[1:bs] - x_pos[0:bs-1]
        y_vel = y_pos[1:bs] - y_pos[0:bs-1]
        x_hat_vel = x_pos_hat[1:bs] - x_pos_hat[0:bs-1]
        y_hat_vel = y_pos_hat[1:bs] - y_pos_hat[0:bs-1]

        # compute velocity loss
        vel_loss  = self.criterion(x_vel, x_hat_vel)
        vel_loss += self.criterion(y_vel, y_hat_vel)

        ### second order loss  ---------------------------------------------------
        # compute the difference in time
        x_acc = x_vel[1:bs-1] - x_vel[0:bs-2]
        y_acc = y_vel[1:bs-1] - y_vel[0:bs-2]
        x_hat_acc = x_hat_vel[1:bs-1] - x_hat_vel[0:bs-2]
        y_hat_acc = y_hat_vel[1:bs-1] - y_hat_vel[0:bs-2]

        # compute acc loss
        acc_loss  = self.criterion(x_acc, x_hat_acc)
        acc_loss += self.criterion(y_acc, y_hat_acc)

        return pos_loss, vel_loss, acc_loss

    def lms_vel_loss(self, heat, heat_hat, only_eye=False, only_mouth=False):
        bs = heat.size(0)
        # get position of lms
        if only_eye:
            heat_     = heat[:,36:48]
            heat_hat_ = heat_hat[:,36:48]
        elif only_mouth:
            mouth     = [51,57,62,66] # middle mouth lms
            heat_     = heat[:,mouth]
            heat_hat_ = heat_hat[:,mouth]
            # heat_     = heat[:,48:68]
            # heat_hat_ = heat_hat[:,48:68]
        else:
            heat_     = heat
            heat_hat_ = heat_hat

        x_pos    , y_pos     = self.pose_extract(heat_)
        x_pos_hat, y_pos_hat = self.pose_extract(heat_hat_)

        # compute the difference in time
        x_vel = x_pos[1:bs] - x_pos[0:bs-1]
        y_vel = y_pos[1:bs] - y_pos[0:bs-1]
        x_hat_vel = x_pos_hat[1:bs] - x_pos_hat[0:bs-1]
        y_hat_vel = y_pos_hat[1:bs] - y_pos_hat[0:bs-1]

        # compute velocity loss
        vel_loss  = self.criterion(x_vel, x_hat_vel)
        vel_loss += self.criterion(y_vel, y_hat_vel)
        return vel_loss
