"""
Learned rounding converter for FP8 and INT8 quantization.

This module implements advanced quantization using learned adaptive rounding
with SVD-based optimization. Inherits from BaseLearnedConverter.
"""
import gc
import math
import torch
from typing import Tuple, Optional, Dict
from tqdm import tqdm
from torch.optim import AdamW, RAdam

from ..constants import (
    TARGET_FP8_DTYPE,
    TARGET_INT8_DTYPE,
    COMPUTE_DTYPE,
    SCALE_DTYPE,
    FP8_MAX,
    INT8_SYMMETRIC_MAX,
)
from ..comfy.quant_ops import BlockWiseINT8Layout
from ..pinned_transfer import transfer_to_gpu_pinned
from ..utils.logging import info, verbose, debug, minimal
from .base_converter import BaseLearnedConverter

class LearnedRoundingConverter(BaseLearnedConverter):
    """
    Learned rounding converter for FP8 and INT8 quantization.

    Inherits shared infrastructure from BaseLearnedConverter.
    Adds format-specific: target_format, scaling_mode, block_size.
    """

    def __init__(
        self,
        scaling_mode: str = "tensor",
        block_size: int = 64,
        target_format: str = "fp8",
        lr: float = 8.077300000003e-3,
        extract_lora: bool = False,
        lora_rank: int = 32,
        lora_depth: int = 1,
        lora_target: Optional[str] = None,
        lora_ar_threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            lr=lr,
            extract_lora=extract_lora,
            lora_rank=lora_rank,
            lora_depth=lora_depth,
            lora_target=lora_target,
            lora_ar_threshold=lora_ar_threshold,
            **kwargs,
        )

        self.block_size = block_size
        self.target_format = target_format
        # Memory Safety Threshold: ~400MB. Gemma 3 IT Embeddings trigger this.
        self.mem_threshold = 100_000_000 

        if target_format == "int8" and scaling_mode not in ("tensor", "block"):
            scaling_mode = "block"
        if scaling_mode == "block3d":
            scaling_mode = "block"
        self.scaling_mode = scaling_mode

        if self.target_format == "int8":
            self.target_dtype = TARGET_INT8_DTYPE
            self.f8_max_val = None
        else:
            self.target_dtype = TARGET_FP8_DTYPE
            self.f8_max_val = FP8_MAX

    def _compute_loss_and_grad(self, current_dq, W_float32, U_k, Vh_k):
        """Memory-aware loss/grad calculation. Offloads to CPU if tensor is massive."""
        if W_float32.numel() > self.mem_threshold:
            # Workspace offloaded to CPU to save ~12GB VRAM on massive layers
            # Use non-blocking transfers to allow some compute overlap
            cpu_dq = current_dq.to("cpu", non_blocking=True)
            cpu_orig = W_float32.to("cpu", non_blocking=True)
            cpu_U = U_k.to("cpu", non_blocking=True)
            cpu_Vh = Vh_k.to("cpu", non_blocking=True)
            
            error = cpu_dq - cpu_orig
            projected_error = cpu_U.T @ error @ cpu_Vh.T
            loss = torch.linalg.norm(projected_error)
            
            # Compute gradient direction on CPU
            grad_dir = cpu_U @ (projected_error / loss.clamp_min(1e-20)) @ cpu_Vh
            
            # Explicit cleanup for large CPU tensors
            del cpu_dq, cpu_orig, cpu_U, cpu_Vh, error, projected_error
            
            return loss.to(self.device), grad_dir.to(self.device)
        else:
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)
            grad_dir = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k
            return loss, grad_dir

    def _optimize_adamw(self, W_float32, scale, U_k, Vh_k):
        W_scaled = W_float32 * scale
        W_rounded = W_scaled.to(self.target_dtype).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        optimizer = AdamW([delta], lr=self.lr)

        best_loss = float("inf")
        best_delta = delta.detach().clone()
        
        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing (AdamW)", leave=False, dynamic_ncols=True)
        for i in pbar:
            optimizer.zero_grad()
            current_dq = (W_rounded + delta) / scale
            
            # Manual backprop to bypass autograd graph memory
            loss, grad_dir = self._compute_loss_and_grad(current_dq, W_float32, U_k, Vh_k)
            
            delta.grad = grad_dir / scale
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_delta = delta.detach().clone()
            
            pbar.set_postfix({"loss": f"{loss.item():.3e}", "best": f"{best_loss:.3e}"})
            
            # Periodic Cache Flush
            if i % 25 == 0 and W_float32.numel() > self.mem_threshold:
                torch.cuda.empty_cache()
                gc.collect()

        pbar.close()
        return W_rounded + best_delta

    def _optimize_radam(self, W_float32, scale, U_k, Vh_k):
        W_rounded = (W_float32 * scale).to(self.target_dtype).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        optimizer = RAdam([delta], lr=self.lr)
        best_loss = float("inf")
        best_delta = delta.detach().clone()

        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing (RAdam)", leave=False)
        for i in pbar:
            optimizer.zero_grad()
            current_dq = (W_rounded + delta) / scale
            loss, grad_dir = self._compute_loss_and_grad(current_dq, W_float32, U_k, Vh_k)
            delta.grad = grad_dir / scale
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_delta = delta.detach().clone()
        pbar.close()
        return W_rounded + best_delta

    def _optimize_prodigy(self, W_float32, scale, U_k, Vh_k):
        from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
        W_rounded = (W_float32 * scale).to(self.target_dtype).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        optimizer = ProdigyPlusScheduleFree([delta], lr=self.lr, use_schedulefree=False, use_speed=self.use_speed)

        best_loss = float("inf")
        best_delta = delta.detach().clone()

        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing (Prodigy)", leave=False)
        for i in pbar:
            optimizer.zero_grad()
            current_dq = (W_rounded + delta) / scale
            loss, grad_dir = self._compute_loss_and_grad(current_dq, W_float32, U_k, Vh_k)
            delta.grad = grad_dir / scale
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_delta = delta.detach().clone()
        pbar.close()
        return W_rounded + best_delta

    def _optimize_original(self, W_float32, scale, U_k, Vh_k):
        W_rounded = (W_float32 * scale).to(self.target_dtype).to(COMPUTE_DTYPE)
        W_q_refined = W_rounded.clone()
        best_loss = float("inf")
        best_tensor = None
        curr_lr = self.lr

        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing (Original)", leave=False)
        for i in pbar:
            with torch.no_grad():
                current_dq = W_q_refined / scale
                loss, grad_dir = self._compute_loss_and_grad(current_dq, W_float32, U_k, Vh_k)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_tensor = W_q_refined.clone()
                
                W_q_refined -= curr_lr * (grad_dir * scale)
                
            if i % 50 == 0 and W_float32.numel() > self.mem_threshold:
                torch.cuda.empty_cache()

        pbar.close()
        return best_tensor if best_tensor is not None else W_q_refined

    def convert(self, W_orig: torch.Tensor, key: Optional[str] = None, depth: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        W_float32 = transfer_to_gpu_pinned(W_orig, self.device, COMPUTE_DTYPE)

        if torch.all(W_float32 == 0):
            quantized_tensor = torch.zeros_like(W_float32, dtype=self.target_dtype)
            dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
            return quantized_tensor, dequant_scale, torch.zeros_like(W_float32), {}

        if self.target_format == "int8":
            if self.scaling_mode == "tensor":
                qdata, scale, dequantized = self._convert_int8_tensorwise(W_float32)
            else:
                qdata, scale, dequantized = self._convert_int8(W_float32)
        else:
            if self.scaling_mode == "row":
                qdata, scale, dequantized = self._convert_fp8_rowwise(W_float32)
            elif self.scaling_mode in ("block", "block2d"):
                qdata, scale, dequantized = self._convert_fp8_block2d(W_float32)
            else:
                qdata, scale, dequantized = self._convert_fp8(W_float32)

        extra_tensors = {}
        if self._should_extract_lora(key, W_orig.shape, depth):
            lora_data = self._extract_error_lora(W_float32, dequantized)
            if lora_data: extra_tensors.update(lora_data)

        return qdata, scale, dequantized, extra_tensors

    def _convert_int8(self, W_float32):
        M, N = W_float32.shape
        qdata, layout_params = BlockWiseINT8Layout.quantize(W_float32, block_size=self.block_size, is_weight=True)
        scale = layout_params["scale"]

        if not self.no_learned_rounding and self.num_iter > 0:
            qdata, scale = self._optimize_int8_learned_rounding(W_float32, qdata, scale)

        dequantized_weight = BlockWiseINT8Layout.dequantize(qdata, scale, self.block_size, is_weight=True, orig_dtype=COMPUTE_DTYPE)
        
        del W_float32
        gc.collect()
        if self.device == "cuda": torch.cuda.empty_cache()
        return qdata, scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized_weight

    def _convert_int8_tensorwise(self, W_float32):
        from ..comfy.quant_ops import TensorWiseINT8Layout
        qdata, layout_params = TensorWiseINT8Layout.quantize(W_float32, is_weight=True)
        scale = layout_params["scale"]

        if not self.no_learned_rounding and self.num_iter > 0:
            qdata, scale = self._optimize_int8_tensorwise_learned_rounding(W_float32, qdata, scale)

        dequantized_weight = TensorWiseINT8Layout.dequantize(qdata, scale, orig_dtype=COMPUTE_DTYPE)
        
        del W_float32
        gc.collect()
        if self.device == "cuda": torch.cuda.empty_cache()
        return qdata, scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized_weight

    def _optimize_int8_tensorwise_learned_rounding(self, W_float32, qdata, scale):
        U_k, Vh_k, k = self._compute_svd_components(W_float32)
        scale_fp8_style = 1.0 / scale.clamp_min(1e-12)
        
        orig_dtype = self.target_dtype
        orig_max = self.f8_max_val
        self.target_dtype = TARGET_INT8_DTYPE
        self.f8_max_val = float(INT8_SYMMETRIC_MAX)

        if self.optimizer_choice == "original":
            final_tensor_scaled = self._optimize_original(W_float32, scale_fp8_style, U_k, Vh_k)
        elif self.optimizer_choice == "adamw":
            final_tensor_scaled = self._optimize_adamw(W_float32, scale_fp8_style, U_k, Vh_k)
        else:
            final_tensor_scaled = self._optimize_radam(W_float32, scale_fp8_style, U_k, Vh_k)

        self.target_dtype = orig_dtype
        self.f8_max_val = orig_max

        with torch.no_grad():
            final_qdata = final_tensor_scaled.clamp(-127, 127).round().to(TARGET_INT8_DTYPE)
        
        self._cleanup_tensors(U_k, Vh_k)
        return final_qdata, scale

    def _int8_dequantize_blockwise(self, qdata, scale, M, N, block_size):
        q_blocked = qdata.reshape(M // block_size, block_size, N // block_size, block_size).permute(0, 2, 1, 3)
        dequantized = q_blocked * scale.unsqueeze(-1).unsqueeze(-1)
        return dequantized.permute(0, 2, 1, 3).reshape(M, N)

    def _optimize_int8_learned_rounding(self, W_float32, qdata, scale):
        U_k, Vh_k, k = self._compute_svd_components(W_float32)
        if self.optimizer_choice == "original":
            final_qdata = self._optimize_int8_original(W_float32, qdata, scale, U_k, Vh_k)
        elif self.optimizer_choice == "adamw":
            final_qdata = self._optimize_int8_adamw(W_float32, qdata, scale, U_k, Vh_k)
        else:
            final_qdata = self._optimize_int8_radam(W_float32, qdata, scale, U_k, Vh_k)
        
        self._cleanup_tensors(U_k, Vh_k)
        return final_qdata, scale

    def _finalize_int8_qdata(self, qdata_float):
        with torch.no_grad():
            qdata_float.clamp_(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX).round_()
            final_qdata = qdata_float.to(TARGET_INT8_DTYPE)
        del qdata_float
        gc.collect()
        if self.device == "cuda": torch.cuda.empty_cache()
        return final_qdata

    def _optimize_int8_adamw(self, W_float32, qdata, scale, U_k, Vh_k):
        M, N = W_float32.shape
        qdata_float = qdata.to(COMPUTE_DTYPE)
        delta = torch.zeros_like(qdata_float, requires_grad=True)
        optimizer = AdamW([delta], lr=self.lr)

        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing INT8 (AdamW)", leave=False)
        for i in pbar:
            optimizer.zero_grad()
            current_dq = self._int8_dequantize_blockwise(qdata_float + delta, scale, M, N, self.block_size)
            loss, grad_dir = self._compute_loss_and_grad(current_dq, W_float32, U_k, Vh_k)
            
            grad_scaled = self._int8_dequantize_blockwise(grad_dir, scale, M, N, self.block_size)
            delta.grad = grad_scaled
            optimizer.step()
            
            if i % 20 == 0 and W_float32.numel() > self.mem_threshold:
                torch.cuda.empty_cache()

        pbar.close()
        return self._finalize_int8_qdata(qdata_float + delta.detach())

    def _optimize_int8_original(self, W_float32, qdata, scale, U_k, Vh_k):
        M, N = W_float32.shape
        q_refined = qdata.to(COMPUTE_DTYPE)
        curr_lr = self.lr

        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing INT8 (Original)", leave=False)
        for i in pbar:
            with torch.no_grad():
                current_dq = self._int8_dequantize_blockwise(q_refined, scale, M, N, self.block_size)
                loss, grad_dir = self._compute_loss_and_grad(current_dq, W_float32, U_k, Vh_k)
                grad_scaled = self._int8_dequantize_blockwise(grad_dir, scale, M, N, self.block_size)
                q_refined -= curr_lr * grad_scaled
                
            if i % 20 == 0 and W_float32.numel() > self.mem_threshold:
                torch.cuda.empty_cache()

        pbar.close()
        return self._finalize_int8_qdata(q_refined)

    def _convert_fp8(self, W_float32):
        w_max = W_float32.abs().max()
        scale = self.f8_max_val / w_max.clamp_min_(1e-12)
        
        if self.no_learned_rounding:
            with torch.no_grad():
                W_f8 = (W_float32 * scale).clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
                dequantized = W_f8.to(COMPUTE_DTYPE) / scale
            return W_f8, (1.0/scale).to(SCALE_DTYPE), dequantized

        U_k, Vh_k, k = self._compute_svd_components(W_float32)
        final_tensor_scaled = self._optimize_adamw(W_float32, scale, U_k, Vh_k)
        
        with torch.no_grad():
            W_f8 = final_tensor_scaled.clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
            dequantized = W_f8.to(COMPUTE_DTYPE) / scale

        return W_f8, (1.0/scale).to(SCALE_DTYPE), dequantized

    def _convert_fp8_rowwise(self, W_float32):
        row_max = W_float32.abs().amax(dim=1, keepdim=True)
        quant_scale = self.f8_max_val / row_max.clamp_min_(1e-12)
        
        if self.no_learned_rounding:
            W_f8 = (W_float32 * quant_scale).clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
            return W_f8, (1.0/quant_scale).squeeze().to(SCALE_DTYPE), W_f8.to(COMPUTE_DTYPE)/quant_scale

        U_k, Vh_k, k = self._compute_svd_components(W_float32)
        final_tensor_scaled = self._optimize_adamw(W_float32, quant_scale, U_k, Vh_k)
        W_f8 = final_tensor_scaled.clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
        
        return W_f8, (1.0/quant_scale).squeeze().to(SCALE_DTYPE), W_f8.to(COMPUTE_DTYPE)/quant_scale

    def _convert_fp8_block2d(self, W_float32):
        M, N = W_float32.shape
        bs = self.block_size
        if M % bs != 0 or N % bs != 0: return self._convert_fp8_rowwise(W_float32)

        W_blocked = W_float32.reshape(M // bs, bs, N // bs, bs).permute(0, 2, 1, 3)
        block_max = W_blocked.abs().amax(dim=(2, 3))
        quant_scale = self.f8_max_val / block_max.clamp_min_(1e-12)

        scale_full = quant_scale.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, bs, bs).permute(0, 2, 1, 3).reshape(M, N)
        
        U_k, Vh_k, k = self._compute_svd_components(W_float32)
        final_tensor_scaled = self._optimize_adamw(W_float32, scale_full, U_k, Vh_k)
        
        W_f8 = final_tensor_scaled.clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
        return W_f8, (1.0/quant_scale).to(SCALE_DTYPE), W_f8.to(COMPUTE_DTYPE)/scale_full
