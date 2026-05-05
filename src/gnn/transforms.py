import torch

class GoalFrameTransform:
    def __init__(self, scale=10.0, v_max=2.0, c_max=100):
        self.scale = scale
        self.v_max = v_max
        self.c_max = c_max

    def transform_pose(self, x, y, angle, gx, gy, ga):
        dx, dy = x - gx, y - gy
        sin_t, cos_t = torch.sin(-ga), torch.cos(-ga)
        
        nx = dx * cos_t - dy * sin_t
        ny = dx * sin_t + dy * cos_t
        na = torch.atan2(torch.sin(angle - ga), torch.cos(angle - ga))
        
        return nx / self.scale, ny / self.scale, na / 3.14159

    def transform_velocity(self, vx, vy, va, ga):
        sin_t, cos_t = torch.sin(-ga), torch.cos(-ga)
        nvx = vx * cos_t - vy * sin_t
        nvy = vx * sin_t + vy * cos_t
        
        return nvx / self.v_max, nvy / self.v_max, va / self.v_max
    
    def normalize_context(self, context):
        for i in range(len(context)):
            context[i] = context[i] / self.c_max