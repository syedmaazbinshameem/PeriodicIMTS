import torch
import torch.nn as nn
from torch import Tensor
from einops import *

from layers.GraFITi import GraFITi_layers
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

import logging

# set up a logger at the top of your file (after imports)
logger = logging.getLogger("GraFITiPeriodicDebug")
logger.setLevel(logging.DEBUG)  # log everything
if not logger.hasHandlers():
    handler = logging.FileHandler("grafiti_periodic_debug.log")  # writes to this file
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PeriodicMLPNet(nn.Module):
    """
    Channel-specific learnable periodic function via MLP.
    Uses sinusoidal phase features but learns arbitrary cycle shapes.
    """

    def __init__(
        self,
        n_channels: int,
        hidden_dim: int = 32,
        n_layers: int = 2,
        period: float = 1.0,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.period = period

        # One MLP per channel (simple and explicit)
        self.mlps = nn.ModuleList([
            self._build_mlp(hidden_dim, n_layers)
            for _ in range(n_channels)
        ])

    def _build_mlp(self, hidden_dim: int, n_layers: int):
        layers = []
        in_dim = 2  # sin, cos

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, t: Tensor) -> Tensor:
        """
        t: (B, L)
        returns: (B, L, C)
        """
        B, L = t.shape

        # Phase features (irregular-time friendly)
        phase = 2 * torch.pi * t / self.period
        phi = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
        # phi: (B, L, 2)

        outputs = []
        for c in range(self.n_channels):
            out_c = self.mlps[c](phi)        # (B, L, 1)
            outputs.append(out_c)

        periodic = torch.cat(outputs, dim=-1)  # (B, L, C)
        return periodic



class FourierPeriodNet(nn.Module):
    """
    Channel-specific learnable periodic function.
    Each channel has its own Fourier coefficients.
    """

    def __init__(self, n_freqs: int, n_channels: int):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_channels = n_channels

        # Shared frequencies
        self.freqs = nn.Parameter(torch.randn(n_freqs))

        # Channel-specific amplitudes
        self.amp_sin = nn.Parameter(torch.randn(n_channels, n_freqs))
        self.amp_cos = nn.Parameter(torch.randn(n_channels, n_freqs))

    def forward(self, t: Tensor) -> Tensor:
        """
        t: (B, L)
        returns: (B, L, C)
        """
        B, L = t.shape
        t = t.unsqueeze(-1)                       # (B, L, 1)
        freqs = self.freqs.view(1, 1, -1)         # (1, 1, K)

        angles = 2 * torch.pi * freqs * t         # (B, L, K)

        sin_part = torch.sin(angles)              # (B, L, K)
        cos_part = torch.cos(angles)              # (B, L, K)

        # Apply channel-specific amplitudes
        sin_part = sin_part.unsqueeze(-2) * self.amp_sin  # (B, L, C, K)
        cos_part = cos_part.unsqueeze(-2) * self.amp_cos  # (B, L, C, K)

        periodic = sin_part + cos_part
        periodic = periodic.sum(dim=-1)           # (B, L, C)

        return periodic


class FourierPeriodNetUniChannel(nn.Module):
    """
    Learnable periodic function using Fourier features.
    p(t) = sum_k a_k sin(2π f_k t) + b_k cos(2π f_k t)
    """

    def __init__(self, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs

        self.freqs = nn.Parameter(torch.randn(n_freqs))
        self.amp_sin = nn.Parameter(torch.randn(n_freqs))
        self.amp_cos = nn.Parameter(torch.randn(n_freqs))

    def forward(self, t: Tensor) -> Tensor:
        """
        t: (B, L) timestamps in [0,1]
        returns: (B, L, 1)
        """
        t = t.unsqueeze(-1)  # (B, L, 1)
        freqs = self.freqs.view(1, 1, -1)

        angles = 2 * torch.pi * freqs * t
        sin_part = torch.sin(angles) * self.amp_sin
        cos_part = torch.cos(angles) * self.amp_cos

        periodic = sin_part + cos_part
        periodic = periodic.sum(dim=-1, keepdim=True)  # (B, L, 1)
        return periodic


class Model(nn.Module):
    '''
    - paper: "GraFITi: Graphs for Forecasting Irregularly Sampled Time Series" (AAAI 2024)
    - paper link: https://ojs.aaai.org/index.php/AAAI/article/view/29560
    - code adapted from: https://github.com/yalavarthivk/GraFITi
    '''
    def __init__(
        self,
        configs: ExpConfigs
    ):
        super().__init__()
        self.configs = configs
        self.dim=configs.enc_in
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.attn_head = configs.n_heads # 4
        self.latent_dim = configs.d_model # 128
        self.n_layers = configs.n_layers # 2

        self.period_net = PeriodicMLPNet(n_channels=self.dim) # learnable periodic function

        self.enc = GraFITi_layers.Encoder(self.dim, self.latent_dim, self.n_layers, self.attn_head, self.configs.task_name, self.configs.n_classes)

    def get_extrapolation(self, context_x, context_w, target_x, target_y, exp_stage):
        # context_x = (16, 290)
        # context_w = (16, 290, 10)
        # target_x = (16, 290)
        # target_y = (16, 290, 10)
        context_mask = context_w[:, :, self.dim:] # last 5 columns
        # context_mask = (16, 290, 5)
        X = context_w[:, :, :self.dim] # first 5 columns
        # X = (16, 290, 5)
        X = X*context_mask
        # X = (16, 290, 5)
        context_mask = context_mask + target_y[:,:,self.dim:]
        # context_mask = (16, 290, 5)
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            output, target_U_, target_mask_ = self.enc(context_x, X, context_mask, target_y[:,:,:self.dim], target_y[:,:,self.dim:], exp_stage)
            return output, target_U_, target_mask_
        else:
            raise NotImplementedError

    def convert_data(self,  x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        return x_time, torch.cat([x_vals, x_mask],-1), y_time, torch.cat([y_vals, y_mask],-1)  

    def forward(
        self, 
        x: Tensor,
        x_mark: Tensor = None, 
        x_mask: Tensor = None, 
        y: Tensor = None, 
        y_mark: Tensor = None, 
        y_mask: Tensor = None,
        exp_stage: str = "train", 
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        DEBUG_ONE_BATCH = True
        logger.debug(BATCH_SIZE) # 16
        logger.debug(DEBUG_ONE_BATCH)
        logger.debug(ENC_IN) # channels 5
        logger.debug(x.shape) # 16 x 287 x 5 (B x T x C)
        logger.debug(x[0]) # 287 x 5
        logger.debug(x)

        Y_LEN = self.pred_len
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)

        x_mark = x_mark[:, :, 0]
        y_mark = y_mark[:, :, 0]

        logger.debug(x_mark) # timestamps (B x T)
        logger.debug(x_mark.shape) # (16, 287)
        logger.debug(y_mark) # query timestamps (B x Q)
        logger.debug(y_mark.shape) # (16, 3)

        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            x_zero_padding = torch.zeros_like(y, device=x.device)
            y_zero_padding = torch.zeros_like(x, device=y.device)

            x_new = torch.cat([x, x_zero_padding], dim=1) # (B, T + Q, C) = (16, 290, 5)


            original_shape = x_new.shape
            x_mark_new = torch.cat([x_mark, y_mark], dim=1) # (B, T + Q) = (16, 290)
            x_mask_new = torch.cat([x_mask, x_zero_padding], dim=1) # (B, T + Q, C) = (16, 290, 5)

            y_new = torch.cat([y_zero_padding, y], dim=1) # (B, T + Q, C) = (16, 290, 5)
            y_mark_new = torch.cat([x_mark, y_mark], dim=1) # (B, T + Q) = (16, 290)
            y_mask_new = torch.cat([y_zero_padding, y_mask], dim=1) # (B, T + Q, C) = (16, 290, 5)

            if DEBUG_ONE_BATCH:
                logger.debug("\n=== RAW INPUT ===")
                logger.debug(f"x_new[0, :, :]: {x_new[0, :, :]}")
                logger.debug(f"y_new[0, :, :]: {y_new[0, :, :]}")
                logger.debug(f"x_mask_new[0, :, :]: {x_mask_new[0, :, :]}")
                logger.debug(f"y_mask_new[0, :, :]: {y_mask_new[0, :, :]}")
                logger.debug(f"x_mark_new[0]: {x_mark_new[0]}")

                logger.debug("\n=== SHAPES n STUFF ===")
                logger.debug(f"x_new shape: {x_new.shape}")
                logger.debug(f"y_new shape: {y_new.shape}")
                logger.debug(f"x_mask shape: {x_mask_new.shape}")
                logger.debug(f"y_mask shape: {y_mask_new.shape}")
                logger.debug(f"x_mark_new shape: {x_mark_new.shape}")

            x_y_mask = torch.cat([x_mask, y_mask], dim=1) # (B, T + Q, C) = (16, 290, 5)
            # x_y_mask = torch.cat([x_mask, torch.zeros_like(y_mask)], dim=1)
            logger.debug(f"x_y_mask: {x_y_mask}")
            logger.debug(f"x_y_mask shape: {x_y_mask.shape}")

        elif self.configs.task_name in ["imputation"]:
            x_new = x
            original_shape = x_new.shape
            x_mark_new = x_mark
            x_mask_new = x_mask

            y_new = y
            y_mark_new = y_mark
            y_mask_new = y_mask

            x_y_mask = x_mask + y_mask
        else:
            raise NotImplementedError
        # END adaptor

        # Compute periodic component
        periodic = self.period_net(x_mark_new)  # output = (B, T, C) = (16, 290, 5) input =  x_mark_new: (B, T + Q) = (16, 290)
        # periodic = periodic.repeat(1, 1, ENC_IN)

        # Subtract periodicity ONLY where observed
        x_new_residual = (x_new - periodic) * x_mask_new # (B, T + Q, C) = (16, 290, 5)
        y_new_residual = y_new * y_mask_new # (B, T + Q, C) = (16, 290, 5)

        if DEBUG_ONE_BATCH:
            logger.debug("\n=== PERIODIC COMPONENT ===")
            logger.debug(f"periodic[0]: {periodic[0]}")
            logger.debug(f"x_new_residual[0]: {x_new_residual[0]}")
            logger.debug(f"y_new_residual[0]: {y_new_residual[0]}")

            logger.debug("\n=== PERIODIC COMPONENT SHAPES ===")
            logger.debug(f"periodic: {periodic.shape}")
            logger.debug(f"x_new_residual: {x_new_residual.shape}")
            logger.debug(f"y_new_residual: {y_new_residual.shape}")


        context_x, context_y, target_x, target_y = self.convert_data(x_mark_new, x_new_residual, x_mask_new, y_mark_new, y_new_residual, y_mask_new)
        # context_x, context_y, target_x, target_y = self.convert_data(x_mark_new, x_new, x_mask_new, y_mark_new, y_new, y_mask_new)
        if len(context_y.shape) == 2:
            context_x = context_x.unsqueeze(0) # (B, T + Q) = (16, 290)
            context_y = context_y.unsqueeze(0) # (B, T + Q, 2C) = (16, 290, 10)
            target_x = target_x.unsqueeze(0) # (B, T + Q) = (16, 290)
            target_y = target_y.unsqueeze(0) # (B, T + Q, 2C) = (16, 290, 10)

        if DEBUG_ONE_BATCH:
            logger.debug("\n=== CONTEXT AND TARGETS ===")
            logger.debug(f"context_x[0]: {context_x[0]}")
            logger.debug(f"context_y[0]: {context_y[0]}")
            logger.debug(f"target_x[0]: {target_x[0]}")
            logger.debug(f"target_y[0]: {target_y[0]}")

            logger.debug("\n=== CONTEXT AND TARGETS SHAPES ===")
            logger.debug(f"context_x shape: {context_x.shape}")
            logger.debug(f"context_y shape: {context_y.shape}")
            logger.debug(f"target_x shape: {target_x.shape}")
            logger.debug(f"target_y shape: {target_y.shape}")

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast', "imputation"]:
            # output, target_U_, target_mask_ = self.get_extrapolation(context_x, context_y, target_x, target_y, exp_stage)
            output_residual, target_U_, target_mask_ = self.get_extrapolation(
                context_x, context_y, target_x, target_y, exp_stage
            )
            output_residual = output_residual.squeeze(-1)
            # output_residual = (B, edges, 1) = (16, 299)
            # target_U_ = (B, edges) = (16, 299)
            # target_mask_ = (B, edges) = (16, 299)

            if DEBUG_ONE_BATCH:
                logger.debug("\n=== AFTER EXTRAPOLATION ===")
                logger.debug(f"output_residual[0]: {output_residual[0]}")
                logger.debug(f"target_U_[0]: {target_U_[0]}")
                logger.debug(f"target_mask_[0]: {target_mask_[0]}")

                logger.debug("\n=== AFTER EXTRAPOLATION SHAPES ===")
                logger.debug(f"output_residual shape: {output_residual.shape}")
                logger.debug(f"target_U_ shape: {target_U_.shape}")
                logger.debug(f"target_mask_ shape: {target_mask_.shape}")

            if exp_stage in ["train", "val"]:
                pred_residual_full = self.unpad_and_reshape(
                    output_residual,   # (B, N_nodes) + (16, 299)
                    x_y_mask,          # (B, T+Q, C) = (16, 290, 5)
                    original_shape     # (B, T+Q, C) = (16, 290, 5)
                    ) 
                # pred_residual_full: (B, T+Q, C) = (16, 290, 5)
                
                if DEBUG_ONE_BATCH:
                    logger.debug("\n=== AFTER UNPAD ===")
                    logger.debug(f"pred_residual_full[0]: {pred_residual_full[0]}")

                    logger.debug("\n=== AFTER UNPAD SHAPES ===")
                    logger.debug(f"pred_residual_full shape: {pred_residual_full.shape}")

                pred_full = pred_residual_full + periodic

                flattenend_pred = self.pad_and_flatten(
                    pred_full,   # (B, T+Q, C) = (16, 290, 5)
                    x_y_mask,          # (B, T+Q, C) = (16, 290, 5)
                ) 

                # flattenend_pred: (B, N_nodes) = (16, 299)

                if DEBUG_ONE_BATCH:
                    logger.debug("\n=== FINAL PRED ===")
                    logger.debug(f"pred_full[0]: {pred_full[0]}")

                    logger.debug("\n=== FINAL PRED SHAPES ===")
                    logger.debug(f"pred_full shape: {pred_full.shape}")

                return {
                    "pred": flattenend_pred,
                    "true": target_U_,
                    "mask": target_mask_
                }
            else:
                pred_residual_full = self.unpad_and_reshape(
                    output_residual,   # (B, N_nodes) = (16, 299)
                    x_y_mask,          # (B, T+Q, C) = (16, 290, 5)
                    original_shape     # (B, T+Q, C) = (16, 290, 5)
                )                       # -> (B, T+Q, C)

                # pred_residual_full: (B, T+Q, C) = (16, 290, 5)

                pred_full = pred_residual_full + periodic        # (B, T+Q, C)

                if DEBUG_ONE_BATCH:
                    logger.debug("\n=== AFTER UNPAD + PERIODIC ADD ===")
                    logger.debug(f"pred_residual_full shape: {pred_residual_full.shape}")
                    logger.debug(f"periodic shape: {periodic.shape}")
                    logger.debug(f"pred_full shape: {pred_full.shape}")

                # 4) Return prediction horizon
                f_dim = -1 if self.configs.features == 'MS' else 0
                PRED_LEN = y.shape[1]

                return {
                    "pred": pred_full[:, -PRED_LEN:, f_dim:],
                    "true": y[:, :, f_dim:],
                    "mask": y_mask[:, :, f_dim:]
                }

        else:
            raise NotImplementedError

    # convert the output back to original shape, to align with api
    def unpad_and_reshape(self, target_U: Tensor, mask: Tensor, original_shape: Tensor):
        batch_size, time_length, ndims = original_shape
        result = torch.zeros(original_shape, dtype=target_U.dtype, device=target_U.device)

        for i in range(batch_size):
            masked_indices = mask[i].view(-1).nonzero(as_tuple=True)[0]
            unpadded_sequence = target_U[i][:len(masked_indices)]
            result[i].view(-1)[masked_indices] = unpadded_sequence
            
        return result


    def pad_and_flatten(
        self,
        dense: torch.Tensor,   # (B, T_total, C)
        mask: torch.Tensor     # (B, T_total, C)
    ) -> torch.Tensor:
        """
        Convert dense (time, channel) tensor into GraFITi node representation.

        Returns:
            flat: (B, N_nodes)
        """
        B, T, C = dense.shape
        flat_list = []

        for i in range(B):
            dense_flat = dense[i].view(-1)            # (T * C,)
            mask_flat = mask[i].view(-1).bool()       # (T * C,)

            # Select only observed entries
            nodes = dense_flat[mask_flat]             # (N_nodes_i,)

            flat_list.append(nodes)

        # Pad to max length across batch
        max_nodes = max(x.shape[0] for x in flat_list)
        out = dense.new_zeros((B, max_nodes))

        for i, nodes in enumerate(flat_list):
            out[i, :nodes.shape[0]] = nodes

        return out
