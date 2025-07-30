import torch
import torch.nn as nn
import torch.nn.functional as F


def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model


# class MimicLoss(nn.Module):  #MSE
#     def __init__(self, channels_s, channels_t):
#         super(MimicLoss, self).__init__()
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.mse = nn.MSELoss()
#
#     def forward(self, y_s, y_t):
#         """Forward computation.
#         Args:
#             y_s (list): The student model prediction with
#                 shape (N, C, H, W) in list.
#             y_t (list): The teacher model prediction with
#                 shape (N, C, H, W) in list.
#         Return:
#             torch.Tensor: The calculated loss value of all stages.
#         """
#         assert len(y_s) == len(y_t)
#         losses = []
#         for idx, (s, t) in enumerate(zip(y_s, y_t)):
#             assert s.shape == t.shape
#             losses.append(self.mse(s, t))
#         loss = sum(losses)
#         return loss


# class MimicLoss(nn.Module):   #KL
#     def __init__(self, channels_s, channels_t, tau=1.0):
#         super(MimicLoss, self).__init__()
#         self.tau = tau
#         self.kl_div = nn.KLDivLoss(reduction='batchmean')  # Use KL divergence loss
#
#     def forward(self, y_s, y_t):
#         """Forward computation for Mimic loss with KL divergence.
#
#         Args:
#             y_s (list): The student model prediction with
#                 shape (N, C, H, W) in list.
#             y_t (list): The teacher model prediction with
#                 shape (N, C, H, W) in list.
#
#         Return:
#             torch.Tensor: The calculated loss value of all stages.
#         """
#         assert len(y_s) == len(y_t)
#         losses = []
#
#         # Use KL divergence instead of MSE
#         for idx, (s, t) in enumerate(zip(y_s, y_t)):
#             assert s.shape == t.shape
#             N, C, H, W = s.shape
#
#             # Normalize predictions using softmax
#             softmax_s = F.log_softmax(s.view(N, C, -1) / self.tau, dim=-1)  # [N, C, H*W]
#             softmax_t = F.softmax(t.view(N, C, -1) / self.tau, dim=-1)  # [N, C, H*W]
#
#             # Compute KL divergence
#             kl_loss = self.kl_div(softmax_s, softmax_t)
#             losses.append(kl_loss)
#
#         # Sum the individual stage losses
#         loss = sum(losses)
#         return loss


# class MimicLoss(nn.Module):   #JS
#     def __init__(self, channels_s, channels_t, tau=1.0):
#         super(MimicLoss, self).__init__()
#         self.tau = tau
#         self.kl_div = nn.KLDivLoss(reduction='batchmean')  # Use KL divergence loss
#
#     def js_divergence(self, p, q):
#         """Compute the Jensen-Shannon divergence between two distributions."""
#         # M = 0.5 * (P + Q)
#         m = 0.5 * (p + q)
#
#         # Compute KL divergence: D_{KL}(P || M) and D_{KL}(Q || M)
#         kl_pm = self.kl_div(p, m)
#         kl_qm = self.kl_div(q, m)
#
#         # JS divergence is the average of the two KL divergences
#         return 0.5 * (kl_pm + kl_qm)
#
#     def forward(self, y_s, y_t):
#         """Forward computation for Mimic loss with JS divergence.
#
#         Args:
#             y_s (list): The student model prediction with
#                 shape (N, C, H, W) in list.
#             y_t (list): The teacher model prediction with
#                 shape (N, C, H, W) in list.
#
#         Return:
#             torch.Tensor: The calculated loss value of all stages.
#         """
#         assert len(y_s) == len(y_t)
#         losses = []
#
#         # Use JS divergence instead of MSE
#         for idx, (s, t) in enumerate(zip(y_s, y_t)):
#             assert s.shape == t.shape
#             N, C, H, W = s.shape
#
#             # Normalize predictions using softmax
#             softmax_s = F.log_softmax(s.view(N, C, -1) / self.tau, dim=-1)  # [N, C, H*W]
#             softmax_t = F.softmax(t.view(N, C, -1) / self.tau, dim=-1)  # [N, C, H*W]
#
#             # Compute JS divergence
#             js_loss = self.js_divergence(softmax_s, softmax_t)
#             losses.append(js_loss)
#
#         # Sum the individual stage losses
#         loss = sum(losses)
#         return loss


class MimicLoss(nn.Module):   #HW-JS
    def __init__(self, channels_s, channels_t, tau=1.0):
        super(MimicLoss, self).__init__()
        self.tau = tau
        self.kl_div = nn.KLDivLoss(reduction='batchmean')  # KL divergence loss

    def js_divergence(self, p, q):
        """Compute the Jensen-Shannon divergence between two distributions."""
        # M = 0.5 * (P + Q)
        m = 0.5 * (p + q)

        # Compute KL divergence: D_{KL}(P || M) and D_{KL}(Q || M)
        kl_pm = self.kl_div(p, m)
        kl_qm = self.kl_div(q, m)

        # JS divergence is the average of the two KL divergences
        return 0.5 * (kl_pm + kl_qm)

    def forward(self, y_s, y_t):
        """Forward computation for HW-JS loss.

        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.

        Return:
            torch.Tensor: The calculated HW-JS loss value.
        """
        assert len(y_s) == len(y_t)
        js_losses = []

        # Calculate JS divergence for each layer
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape

            # Normalize predictions using softmax
            softmax_s = F.log_softmax(s.view(N, C, -1) / self.tau, dim=-1)  # [N, C, H*W]
            softmax_t = F.softmax(t.view(N, C, -1) / self.tau, dim=-1)  # [N, C, H*W]

            # Compute JS divergence for this layer
            js_loss = self.js_divergence(softmax_s, softmax_t)
            js_losses.append(js_loss)

        # Convert list of JS losses to a tensor
        js_losses = torch.stack(js_losses)

        # Calculate the weight for each layer based on JS loss (softmax normalization)
        weights = F.softmax(js_losses, dim=0)  # Use softmax for weight normalization

        # Compute the final weighted HW-JS loss
        total_loss = torch.sum(weights * js_losses)

        return total_loss
class CWDLoss(nn.Module):      #KL散度
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """

    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape

            N, C, H, W = s.shape

            # normalize in channel diemension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)  # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss


class CWDLoss(nn.Module):   # HW-JS
    """PyTorch implementation of Hierarchically Weighted Jensen-Shannon Divergence for Knowledge Distillation."""

    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau
        self.channels_s = channels_s
        self.channels_t = channels_t

    def js_divergence(self, p, q):
        """Calculate the Jensen-Shannon Divergence between two distributions."""
        m = 0.5 * (p + q)
        p_log = F.log_softmax(p, dim=1)
        q_log = F.log_softmax(q, dim=1)
        m_log = F.log_softmax(m, dim=1)
        return 0.5 * (F.kl_div(p_log, m_log, reduction='batchmean') +
                      F.kl_div(q_log, m_log, reduction='batchmean'))

    def forward(self, y_s, y_t):
        """Forward computation for HW-JS loss.

        Args:
            y_s (list): The student model predictions, shape (N, C, H, W).
            y_t (list): The teacher model predictions, shape (N, C, H, W).

        Returns:
            torch.Tensor: The HW-JS loss value.
        """
        assert len(y_s) == len(y_t)
        js_losses = []

        # Compute JS loss for each layer and store
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape

            # Normalize predictions using softmax
            softmax_s = F.softmax(s.view(N, C, -1) / self.tau, dim=-1)  # [N, C, H*W]
            softmax_t = F.softmax(t.view(N, C, -1) / self.tau, dim=-1)  # [N, C, H*W]

            # Compute JS divergence
            js_loss = self.js_divergence(softmax_s, softmax_t)
            js_losses.append(js_loss)

        # Convert list to tensor
        js_losses = torch.stack(js_losses)

        # Compute the weight for each layer based on JS loss
        weights = F.softmax(js_losses, dim=0)  # Use softmax normalization

        # Compute the final weighted HW-JS loss
        total_loss = torch.sum(weights * js_losses)

        return total_loss

# class CWDLoss(nn.Module):    #均方误差
#     """PyTorch version of Channel-wise Distillation Loss with MSE for Semantic Segmentation."""
#
#     def __init__(self, channels_s, channels_t, tau=1.0):
#         super(CWDLoss, self).__init__()
#         self.tau = tau
#
#     def forward(self, y_s, y_t):
#         """Forward computation.
#         Args:
#             y_s (list): The student model prediction with
#                 shape (N, C, H, W) in list.
#             y_t (list): The teacher model prediction with
#                 shape (N, C, H, W) in list.
#         Return:
#             torch.Tensor: The calculated loss value of all stages.
#         """
#         assert len(y_s) == len(y_t)
#         losses = []
#
#         for idx, (s, t) in enumerate(zip(y_s, y_t)):
#             assert s.shape == t.shape
#
#             # Compute the Mean Squared Error (MSE)
#             loss = torch.mean((s - t) ** 2)  # Calculate the MSE between student and teacher predictions
#             losses.append(loss)
#
#         # Return the average MSE loss over all stages
#         loss = sum(losses) / len(losses)
#         return loss

# class CWDLoss(nn.Module):   #概率分布的均方误差
#     """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
#     <https://arxiv.org/abs/2011.13256>`_.
#     """
#
#     def __init__(self, channels_s, channels_t, tau=1.0):
#         super(CWDLoss, self).__init__()
#         self.tau = tau
#
#     def forward(self, y_s, y_t):
#         """Forward computation.
#         Args:
#             y_s (list): The student model prediction with
#                 shape (N, C, H, W) in list.
#             y_t (list): The teacher model prediction with
#                 shape (N, C, H, W) in list.
#         Return:
#             torch.Tensor: The calculated loss value of all stages.
#         """
#         assert len(y_s) == len(y_t)
#         losses = []
#
#         for idx, (s, t) in enumerate(zip(y_s, y_t)):
#             assert s.shape == t.shape
#
#             N, C, H, W = s.shape
#
#             # Normalize the predictions to get probability distributions
#             softmax_pred_s = F.softmax(s.view(-1, H * W) / self.tau, dim=1)  # [N*C, H*W]
#             softmax_pred_t = F.softmax(t.view(-1, H * W) / self.tau, dim=1)  # [N*C, H*W]
#
#             # Compute Mean Squared Error (MSE) between the predicted distributions
#             mse_loss = torch.mean((softmax_pred_s - softmax_pred_t) ** 2)
#
#             losses.append(mse_loss)
#
#         loss = sum(losses)
#
#         return loss



# class CWDLoss(nn.Module):   #JS散度
#     """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
#     <https://arxiv.org/abs/2011.13256>`_.
#     """
#
#     def __init__(self, channels_s, channels_t, tau=1.0):
#         super(CWDLoss, self).__init__()
#         self.tau = tau
#
#     def forward(self, y_s, y_t):
#         """Forward computation.
#         Args:
#             y_s (list): The student model prediction with
#                 shape (N, C, H, W) in list.
#             y_t (list): The teacher model prediction with
#                 shape (N, C, H, W) in list.
#         Return:
#             torch.Tensor: The calculated loss value of all stages.
#         """
#         assert len(y_s) == len(y_t)
#         losses = []
#
#         for idx, (s, t) in enumerate(zip(y_s, y_t)):
#             assert s.shape == t.shape
#
#             N, C, H, W = s.shape
#
#             # Normalize the predictions to get probability distributions
#             softmax_pred_s = F.softmax(s.view(-1, H * W) / self.tau, dim=1)  # [N*C, H*W]
#             softmax_pred_t = F.softmax(t.view(-1, H * W) / self.tau, dim=1)  # [N*C, H*W]
#
#             # Compute the mixture distribution M
#             M = 0.5 * (softmax_pred_s + softmax_pred_t)
#
#             # Compute the KL divergence terms
#             kl_p_m = torch.sum(softmax_pred_s * (torch.log(softmax_pred_s + 1e-10) - torch.log(M + 1e-10)), dim=1)
#             kl_q_m = torch.sum(softmax_pred_t * (torch.log(softmax_pred_t + 1e-10) - torch.log(M + 1e-10)), dim=1)
#
#             # Compute the JS Divergence as the average of both KL divergences
#             js_loss = 0.5 * (kl_p_m + kl_q_m)
#
#             # Take the mean over all pixels and batches
#             losses.append(torch.mean(js_loss))
#
#         loss = sum(losses)
#
#         return loss


class MGDLoss(nn.Module):
    def __init__(self, channels_s, channels_t, alpha_mgd=0.00002, lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = [
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)).to(device) for channel in channels_t
        ]

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)
        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss


class Distill_LogitLoss:
    def __init__(self, p, t_p, alpha=0.25):
        t_ft = torch.cuda.FloatTensor if t_p[0].is_cuda else torch.Tensor
        self.p = p
        self.t_p = t_p
        self.logit_loss = t_ft([0])
        self.DLogitLoss = nn.MSELoss(reduction="none")
        self.bs = p[0].shape[0]
        self.alpha = alpha

    def __call__(self):
        # per output
        assert len(self.p) == len(self.t_p)
        for i, (pi, t_pi) in enumerate(zip(self.p, self.t_p)):  # layer index, layer predictions
            assert pi.shape == t_pi.shape
            self.logit_loss += torch.mean(self.DLogitLoss(pi, t_pi))
        return self.logit_loss[0] * self.alpha


def get_fpn_features(x, model, fpn_layers=[15, 18, 21]):
    y, fpn_feats = [], []
    with torch.no_grad():
        model = de_parallel(model)
        module_list = model.model[:-1] if hasattr(model, "model") else model[:-1]
        for m in module_list:
            # if not from previous layer
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)
            y.append(x if m.i in model.save else None)  # save output
            if m.i in fpn_layers:
                fpn_feats.append(x)
    return fpn_feats



def get_channels(model, fpn_layers=[15, 18, 21]):
    y, out_channels = [], []
    p = next(model.parameters())
    x = torch.zeros((1, 3, 64, 64), device=p.device)
    with torch.no_grad():
        model = de_parallel(model)
        module_list = model.model[:-1] if hasattr(model, "model") else model[:-1]

        for m in module_list:
            # if not from previous layer
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)
            y.append(x if m.i in model.save else None)  # save output
            if m.i in fpn_layers:
                out_channels.append(x.shape[1])
    return out_channels


class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='cwd'):
        super(FeatureLoss, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList([
            nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1, padding=0).to(device)
            for channel, tea_channel in zip(channels_s, channels_t)
        ])
        self.norm = [
            nn.BatchNorm2d(tea_channel, affine=False).to(device)
            for tea_channel in channels_t
        ]

        if distiller == 'mimic':
            self.feature_loss = MimicLoss(channels_s, channels_t)

        elif distiller == 'mgd':
            self.feature_loss = MGDLoss(channels_s, channels_t)

        elif distiller == 'cwd':
            self.feature_loss = CWDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[idx](s)
            s = self.norm[idx](s)
            t = self.norm[idx](t)
            tea_feats.append(t)
            stu_feats.append(s)

        loss = self.feature_loss(stu_feats, tea_feats)
        return loss