import torch
from torch import nn, math
from torch.nn import RMSNorm
import torch.nn.functional as F
from typing import Tuple, List, Optional

from .config_schema import SabdaConfig
from .utils import _normalize_axes, _str_to_dtype, get_activation_fn


class DenseGeneral(nn.Module):
    """
    PyTorch equivalent of flax.linen.DenseGeneral.
    Stores weights (`kernel`) in the same layout as Jax and uses torch.tensordot
    for the generalized matrix multiplication. Weight/bias shapes are calculated
    and parameters created during initialization based on config.
    `load_weights` validates shapes and copies data.

    Attributes:
    in_shapes (Tuple[int, ...]): Sizes of the input dimensions specified by `axis`.
    out_features (Tuple[int, ...]): Shape of the output features (non-contracted dims).
    axis (Tuple[int, ...]): input axis or axes to contract
    weight (nn.Parameter): The kernel parameters
    """

    def __init__(
        self,
        in_shapes: Tuple[int, ...],
        out_features: Tuple[int, ...],
        axis: Tuple[int, ...] = (-1,),
        weight_dtype: Optional[torch.dtype] = None,
    ) -> None:
            
        super().__init__()
        self.in_shapes = in_shapes
        self.out_features = out_features
        self.axis = axis

        self.kernel_shape = self.in_shapes + self.out_features

        # Define the factory_kwargs for the dtype of the weight
        factory_kwargs = {"dtype": weight_dtype}
        self.weight = nn.Parameter(torch.empty(self.kernel_shape, **factory_kwargs))
        self.register_parameter("bias", None)

        # weight inilization
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # weight inilization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Normalize the axis
        norm_axis = _normalize_axes(self.axis, inputs.ndim)

        # The contract kernel axis
        kernel_contract_axes = tuple(range(len(norm_axis)))

        original_dtype = inputs.dtype # Save the original dtype
        output = torch.tensordot(
            inputs.to(self.weight.dtype), # convert input dtype into weight dtype for tensordot
            self.weight,
            dims=(norm_axis, kernel_contract_axes),
        )

        return output.to(original_dtype)

        
class GatedMLPFeedForward(nn.Module):
    """Implement Gated MLP as Feed Forward Block used later in the architecture"""
    def __init__(
        self,
        d_embd: int,
        d_ff: int,
        activations: List[str] = ['silu', 'linear'],
        dropout_rate: float = 0.0,
        weight_dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__()

        self.wi_fused = DenseGeneral(
            in_shapes=(d_embd,),
            out_features=(len(activations), d_ff),
            axis=(-1,),
            weight_dtype=weight_dtype
        )

        if len(activations) != 2:
            raise ValueError("activations in GatedMLPFeedForward currently expects exactly two activation functions for gating.")
        
        self.activation_fn_gate = get_activation_fn(activations[0]) # SiLU
        self.activation_fn_up = get_activation_fn(activations[1]) # Linear

        self.wo = DenseGeneral(
                in_shapes=(d_ff,),
                out_features=(d_embd,),
                axis=(-1,),
                weight_dtype=weight_dtype
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, deterministic: bool = True):
        # Assume the x has been normalized
        fused_hidden = self.wi_fused(x)

        gate_input = fused_hidden[..., 0, :]
        up_input = fused_hidden[..., 1,  :]

        gate_output = self.activation_fn_gate(gate_input)
        up_output = self.activation_fn_up(up_input)

        activated_hidden = gate_output * up_output

        if not deterministic:
            activated_hidden = self.dropout(activated_hidden)
        
        output = self.wo(activated_hidden)
        
        return output


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation in PyTorch"""

    def __init__(
        self,
        d_heads: int,
        min_timescale: int = 1,
        max_timescale: int = 10_000,
        dtype: torch.dtype = torch.float32
    ) -> None:

        super().__init__() 
        if d_heads % 2 != 0:
                raise ValueError("Embedding Dimension must be even number for RoPE")
        
        self.d_heads = d_heads
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.dtype = dtype

        # Calculation for fraction and timescale
        fraction = (2.0 * torch.arange(0, d_heads // 2)) / d_heads

        # Buffer 'timescale' will not be saved in the state_dict model (persistent=False)
        self.register_buffer(
                "timescale",
                self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
        )

    def extra_repr(self) -> str:
         return f"timescale_shape: {self.timescale.shape}"
    

    def forward(self, inputs: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        # Ensure the dimension of position
        position =  position.unsqueeze(-1).unsqueeze(-1)

        # Makesure the timescale have the same device as inputs
        timescale = self.timescale.to(inputs.device)

        sinusoid_inp = position / timescale
        sin = torch.sin(sinusoid_inp).to(inputs.dtype)
        cos = torch.cos(sinusoid_inp).to(inputs.dtype)

        # Separate the input into 2 part
        first_half, second_half = torch.chunk(inputs, 2, dim=-1)

        # Apply the rotation
        first_part = cos * first_half - sin * second_half
        second_part = cos * second_half + sin * first_half

        return torch.cat((first_part, second_part), dim=-1)

class KVCache:
    def __init__(
        self,
        n_heads: int,
        max_len: int,
        d_heads: int,
        device: torch.device,
        batch_size: int, # TAMBAHKAN ini
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None
    ) -> None:
        # self.cache_batch_size = 2 # HAPUS atau abaikan ini
        self.current_idx = 0
        self.max_len = max_len
        self.expected_batch_size = batch_size # Simpan batch size yang diharapkan

        if k is None:
            self.k = torch.zeros((self.expected_batch_size, n_heads, max_len, d_heads), device=device)
        else:
            if k.shape[0] != self.expected_batch_size:
                # Peringatan ini jadi lebih relevan sekarang
                print(f"Warning: KVCache initialized with k.shape[0]={k.shape[0]} but expected batch_size={self.expected_batch_size}")
            self.k = k
        
        if v is None:
            self.v = torch.zeros((self.expected_batch_size, n_heads, max_len, d_heads), device=device)
        else:
            if v.shape[0] != self.expected_batch_size:
                print(f"Warning: KVCache initialized with v.shape[0]={v.shape[0]} but expected batch_size={self.expected_batch_size}")
            self.v = v
    
    # Metode lain (get_kv, update_cache, prefill_kv) tetap sama,
    # tapi pastikan prefill_kv bisa menangani assignment dengan benar.
    # Error di prefill_kv terjadi karena self.k punya batch dim yang salah.
    # Dengan __init__ yang benar, prefill_kv seharusnya aman.
    def prefill_kv(self, k_prompt: torch.Tensor, v_prompt: torch.Tensor) -> None:
        prefill_len = k_prompt.shape[2]

        if prefill_len > self.max_len:
            raise ValueError(f"Prefill length ({prefill_len}) exceeds KVCache max_len ({self.max_len})")

        # Pastikan batch dimension k_prompt sesuai dengan self.k
        if self.k.shape[0] != k_prompt.shape[0]:
            raise ValueError(f"Batch dimension mismatch in prefill_kv: self.k batch is {self.k.shape[0]}, k_prompt batch is {k_prompt.shape[0]}")

        self.k[:, :, :prefill_len, :] = k_prompt
        self.v[:, :, :prefill_len, :] = v_prompt
        
        self.current_idx = prefill_len

    def get_kv_for_attention(self, current_k_step: torch.Tensor, current_v_step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenates historical K/V from cache with the K/V of the current step.
        current_k_step, current_v_step: Tensors for the current step, shape (B, n_heads, 1, d_heads).
        """
        # Ensure current_k_step and current_v_step have the expected shape (B, N, 1, H)
        if not (current_k_step.ndim == 4 and current_k_step.shape[2] == 1 and \
                current_v_step.ndim == 4 and current_v_step.shape[2] == 1):
            raise ValueError(f"current_k_step/v_step must have shape (B, N, 1, H), "
                             f"got k: {current_k_step.shape}, v: {current_v_step.shape}")

        if self.current_idx == 0: # No past history in cache
            return current_k_step, current_v_step
        else:
            past_k = self.k[:, :, :self.current_idx, :]
            past_v = self.v[:, :, :self.current_idx, :]
            
            attn_k = torch.cat((past_k, current_k_step), dim=2) # Concatenate along sequence_length dim
            attn_v = torch.cat((past_v, current_v_step), dim=2)
            return attn_k, attn_v

    def update_cache(self, k_onestep: torch.Tensor, v_onestep: torch.Tensor):
        """
        Adds the K/V for the current single step to the cache.
        k_onestep, v_onestep: Tensors for the current step, shape (B, n_heads, 1, d_heads).
        """
        if not (k_onestep.ndim == 4 and k_onestep.shape[2] == 1 and \
                v_onestep.ndim == 4 and v_onestep.shape[2] == 1):
            raise ValueError(f"k_onestep/v_onestep for update_cache must have shape (B, N, 1, H), "
                             f"got k: {k_onestep.shape}, v: {v_onestep.shape}")

        if self.current_idx >= self.max_len:
            # This condition should ideally be handled by the generation loop stopping criterion
            # or by implementing a sliding window if max_len is exceeded.
            # For now, we raise an error as it implies max_new_tokens might be too large for cache.
            raise ValueError(f"KVCache is full (current_idx: {self.current_idx}, max_len: {self.max_len}). Cannot update.")
        
        self.k[:, :, self.current_idx : self.current_idx + 1, :] = k_onestep
        self.v[:, :, self.current_idx : self.current_idx + 1, :] = v_onestep
        self.current_idx += 1


class Attention(nn.Module):
    def __init__(
        self,
        d_embd_q: int,
        d_embd_kv: int,
        n_query_heads: int,
        n_kv_heads: int,
        d_heads: int,
        d_output: Optional[int] = None,
        dropout_rate: float = 0.0,
        cross_attention: bool = False,
        min_rope_timescale: int = 1,
        max_rope_timescale: int = 10_000,
        weight_dtype: Optional[torch.dtype] = None,
        rope_dtype: Optional[torch.dtype] = torch.float32
    ) -> None:
        super().__init__()

        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.d_heads = d_heads
        self.cross_attention = cross_attention
        self.dropout_rate = dropout_rate

        self.d_output = d_output if d_output is not None else d_embd_q

        if self.n_query_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_query_heads ({self.n_query_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads}) for GQA."
            )

        self.n_gqa_groups = self.n_query_heads // self.n_kv_heads

        # Initialize Projection
        self.W_Q = DenseGeneral(
            in_shapes=(d_embd_q,),
            out_features=(n_query_heads, d_heads),
            axis=(-1,),
            weight_dtype=weight_dtype
        )
        self.W_K = DenseGeneral(
            in_shapes=(d_embd_kv,),
            out_features=(n_kv_heads, d_heads),
            axis=(-1,),
            weight_dtype=weight_dtype
        )
        self.W_V = DenseGeneral(
            in_shapes=(d_embd_kv,),
            out_features=(n_kv_heads, d_heads),
            axis=(-1,),
            weight_dtype=weight_dtype
        )
        self.W_O = DenseGeneral(
            in_shapes=(n_query_heads, d_heads),
            out_features=(self.d_output,),
            axis=(-2,-1),
            weight_dtype=weight_dtype
        )

        # Initialize RoPE
        self.rope = RotaryPositionEmbedding(
            d_heads=d_heads,
            min_timescale=min_rope_timescale,
            max_timescale=max_rope_timescale,
            dtype=rope_dtype
        )

    def forward(
        self,
        X_q: torch.Tensor,
        X_kv: torch.Tensor,
        q_pos: torch.Tensor,
        kv_pos: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        prefill: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Performs attention calculation with optional KV caching.

        Args:
            X_q: Input Query tensor (B, T, d_embd_q).
            X_kv: Input Key/Value tensor (B, S, d_embd_kv).
            q_pos: Positions for queries (B, T).
            kv_pos: Positions for keys/values (B, S). If None, uses q_positions.
            deterministic: If True, disable dropout.
            attn_mask: Attention mask. (B, n_query_heads/1, T, S)
            cache: KVCache.
            prefill: If True, use prefill mode.

        Returns:
            A tuple containing:
            - output: The attention output tensor (B, T, d_output).
            - present_kv: The K/V state to be cached for the next step ((B, N, S_new, H), (B, N, S_new, H)). For self-attn, S_new = S_past + S. For cross-attn, S_new = S_kv.
        """

        # Preparing kv_pos
        if kv_pos is None:
            # For self-attention, kv_pos = q_pos
            kv_pos = q_pos

        original_dtype = X_q.dtype

        # Project Queries and apply RoPE
        # Xq: (B, T, d_embd_q) -> W_Q -> (B, T, n_query_heads, d_heads)
        queries = self.W_Q(X_q)

        # Apply RoPE on Queries
        queries = self.rope(queries, position=q_pos) # queries (B, T, n_query_heads, d_heads), q_pos (B, T)

        # Transpose into (B, n_query_heads, T, d_heads)
        queries = queries.transpose(1, 2) # (Batch, NumHeads, SeqLen, HeadDim)

        # Init variable for Key, Value, and new_kv_cache
        keys: torch.Tensor
        values: torch.Tensor
        new_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Cross Attention vs Self Attention Logic
        if self.cross_attention:
            # For Cross Attention, K/V cache taken from precompute cache (from encoder)
            if cache is None:
                raise ValueError("KVCache must be provided for cross-attention.")
            
            keys = cache.k # Shape: (B_cache, n_kv_heads_cross (already repeated if GQA), S_encoder, d_heads)
            values = cache.v # Shape: (B_cache, n_kv_heads_cross (already repeated if GQA), S_encoder, d_heads)

            # Ensure the dimension of KV's Heads same as Queries (MHA)
            if keys.shape[1] != self.n_query_heads or values.shape[1] != self.n_query_heads:
                raise ValueError(
                    f"Cross-attention cache head dimension ({keys.shape[1]}) "
                    f"does not match num_query_heads ({self.n_query_heads}). "
                    "Cache should be pre-repeated for GQA."
                )
            # Ensure that Batch Dimension is matched if cache B=2 and input B=1
            if queries.shape[0] != keys.shape[0] and keys.shape[0] == 2 and queries.shape[0] == 1:
                # If CFG (B=2 in cache) and inference (B=1 in query), we take conditional part from the cache
                keys = keys[1:2, ...] # Take slice, not indices, ensure the batch dimension still exist
                values = values[1:2, ...]

            elif queries.shape[0] != keys.shape[0]:
                raise ValueError(f"Batch size mismatch between query ({queries.shape[0]}) and K/V cache ({keys.shape[0]}) for cross-attention.")
            
            new_kv_cache = None
        else:
            keys = self.rope(self.W_K(X_kv), position=kv_pos).transpose(1, 2)
            values = self.W_V(X_kv).transpose(1, 2)

            # Handle GQA Logic, Repeat K/V if n_kv_heads < n_query_heads
            if self.n_gqa_groups > 1:
                keys_gqa = keys.repeat_interleave(self.n_gqa_groups, dim=1)
                values_gqa = values.repeat_interleave(self.n_gqa_groups, dim=1)
            else: # MHA
                keys_gqa = keys
                values_gqa = values

            # Cache Management for Self-Attention
            if cache is None: # Training mode or Encoder self-attention (no KVCache object passed)
                keys = keys_gqa # keys_gqa here is the full sequence if training
                values = values_gqa
                new_kv_cache = None
            else: # Decoder self-attention with KVCache object passed
                if prefill: # Mode prefill (e.g., processing an audio prompt during inference)
                    # keys_gqa and values_gqa are from the entire prompt
                    cache.prefill_kv(keys_gqa, values_gqa) 
                    keys = keys_gqa # Use the full prompt K/V for attention
                    values = values_gqa
                    new_kv_cache = None # Cache is prefilled, no new K/V to return for update via this path
                else: # Autoregressive decoding step (prefill=False)
                    # keys_gqa and values_gqa are for the CURRENT SINGLE STEP: (B, n_query_heads, 1, d_heads)
                    
                    # Get full K,V sequence for attention (history + current step)
                    keys, values = cache.get_kv_for_attention(keys_gqa, values_gqa)
                    
                    # The K,V for the current step needs to be returned so it can be added to cache by the caller
                    new_kv_cache = (keys_gqa, values_gqa) 

        # Calculate Attention Scores
        attn_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            dropout_p=self.dropout_rate if not deterministic else 0.0,
            scale=1.0
            )

        # Transpose and Reshape Output Attention
        # (B, n_query_heads, T, d_heads) -> (B, T, n_query_heads, d_heads)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Project the output
        output = self.W_O(attn_output)

        return output.to(original_dtype), new_kv_cache
    

class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_embd: int,
        n_heads: int,
        d_heads: int,
        d_ff: int,
        ff_activations: Tuple[str, ...] = ('silu', 'linear'),
        dropout_rate: float = 0.0,
        norm_eps: float = 1e-5,
        weight_dtype: Optional[torch.dtype] = None,
        min_rope_timescale: int = 1,
        max_rope_timescale: int = 10_000,
        rope_dtype: Optional[torch.dtype] = torch.float32
    ) -> None:
            
        super().__init__()

        self.pre_sa_norm = RMSNorm(d_embd, norm_eps)
        self.self_attention = Attention(
            d_embd_q=d_embd,
            d_embd_kv=d_embd,
            n_query_heads=n_heads,
            n_kv_heads=n_heads,
            d_heads=d_heads,
            d_output=d_embd,
            dropout_rate=dropout_rate,
            min_rope_timescale=min_rope_timescale,
            max_rope_timescale=max_rope_timescale,
            weight_dtype=weight_dtype,
            rope_dtype=rope_dtype
        )

        self.pre_ff_norm = RMSNorm(d_embd, norm_eps)

        self.ff = GatedMLPFeedForward(
            d_embd=d_embd,
            d_ff=d_ff,
            activations=ff_activations,
            dropout_rate=dropout_rate,
            weight_dtype=weight_dtype
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor, # Input from previous layer or embedding, shape (B, T, d_embd)
        src_pos: torch.Tensor, # Posisi for RoPE, shape (B, T)
        attn_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        # Sub-lapisan Self-Attention (dengan Pre-Norm dan koneksi residual)
        residual = x
        x_norm = self.pre_sa_norm(x)

        # Call the self-attention. Cache and prefill doesn't relevant for encoder self-attention.
        sa_output, _ = self.self_attention(
            X_q=x_norm,
            X_kv=x_norm,
            q_pos=src_pos,
            kv_pos=src_pos,
            deterministic=deterministic,
            attn_mask=attn_mask,
            cache=None, # Doesn't use cache for encoder
            prefill=False
        )
        x = residual + sa_output # First residual connection

        # Feed Forward Sub Layer (with Pre-Norm and residual connection)
        residual = x
        x_norm = self.pre_ff_norm(x)
        
        ff_output = self.ff(
            x_norm,
            deterministic=deterministic
        )
        x = residual + ff_output # Second residual connection
        
        if not deterministic:
            x = self.dropout(x)
        return x
    

class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_embd_decoder: int,
        d_embd_encoder: int,
        
        # Self-Attention (GQA) Parameter
        n_gqa_query_heads: int,
        n_gqa_kv_heads: int,
        d_gqa_head: int,
        
        # Cross-Attention (MHA) Parameter
        n_cross_query_heads: int,
        d_cross_head: int,
        
        # FF Parameter
        d_ff: int,
        ff_activations: Tuple[str, ...] = ('silu', 'linear'),
        
        # Global Parameter
        dropout_rate: float = 0.0,
        norm_eps: float = 1e-5,
        
        # Precision and RoPE parameter
        weight_dtype: Optional[torch.dtype] = None,
        min_rope_timescale: int = 1,
        max_rope_timescale: int = 10_000,
        rope_dtype: Optional[torch.dtype] = torch.float32
    ):
        super().__init__()

        # Normalization before Masked Self-Attention (Pre-Norm)
        self.pre_sa_norm = RMSNorm(d_embd_decoder, eps=norm_eps)
        
        # Masked Self-Attention (GQA)
        self.self_attention = Attention(
            d_embd_q=d_embd_decoder,
            d_embd_kv=d_embd_decoder, # Untuk self-attention, q dan kv dari input yang sama
            n_query_heads=n_gqa_query_heads,
            n_kv_heads=n_gqa_kv_heads, # Menggunakan n_gqa_kv_heads untuk GQA
            d_heads=d_gqa_head,
            d_output=d_embd_decoder,
            dropout_rate=dropout_rate,
            cross_attention=False,
            min_rope_timescale=min_rope_timescale,
            max_rope_timescale=max_rope_timescale,
            weight_dtype=weight_dtype,
            rope_dtype=rope_dtype
        )

        # Normalisasi before Cross-Attention (Pre-Norm)
        self.pre_ca_norm = RMSNorm(d_embd_decoder, eps=norm_eps)

        # Cross-Attention (usually MHA)
        self.cross_attention = Attention(
            d_embd_q=d_embd_decoder,
            d_embd_kv=d_embd_encoder,
            n_query_heads=n_cross_query_heads,
            n_kv_heads=n_cross_query_heads, 
            d_heads=d_cross_head,
            d_output=d_embd_decoder,
            dropout_rate=dropout_rate,
            cross_attention=True,
            min_rope_timescale=min_rope_timescale,
            max_rope_timescale=max_rope_timescale,
            weight_dtype=weight_dtype,
            rope_dtype=rope_dtype
        )

        # Normalization before FeedForward (Pre-Norm)
        self.pre_ff_norm = RMSNorm(d_embd_decoder, eps=norm_eps)
        
        # Feed Forward
        self.ff = GatedMLPFeedForward(
            d_embd=d_embd_decoder,
            d_ff=d_ff,
            activations=list(ff_activations),
            dropout_rate=dropout_rate,
            weight_dtype=weight_dtype
        )
    
    def forward(
        self,
        x: torch.Tensor, # Input from previous decoder layer or target embedding, shape (B, T_tgt, d_embd_decoder)
        encoder_output: torch.Tensor,
        
        # Posisi
        tgt_pos: torch.Tensor, # Target position (Query in self-attn & cross-attn), shape (B, T_tgt)
        src_pos: Optional[torch.Tensor], # Source position (Key/Value in cross-attn dari encoder), shape (B, T_src)

        # Mask-mask
        self_attn_mask: Optional[torch.Tensor],     # Mask for self-attention (causal + padding), shape (B, n_gqa_query_heads/1, T_tgt, T_tgt)
        cross_attn_mask: Optional[torch.Tensor],    # Mask for cross-attention (padding for the encoder_output), shape (B, n_cross_query_heads/1, T_tgt, T_src)
        
        # Cache (for inferensce purpose)
        self_attn_kv_cache: Optional[KVCache] = None,
        cross_attn_kv_cache: Optional[KVCache] = None, # Ini akan berisi K/V dari encoder yang sudah di-precompute

        deterministic: bool = True,
        prefill: bool = False # Untuk self-attention cache
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Masked Self-Attention Sub Layer
        residual = x
        x_norm = self.pre_sa_norm(x)
        
        sa_output, new_self_attn_kv_for_cache = self.self_attention(
            X_q=x_norm,
            X_kv=x_norm,
            q_pos=tgt_pos,
            kv_pos=tgt_pos,
            deterministic=deterministic,
            attn_mask=self_attn_mask,
            cache=self_attn_kv_cache,
            prefill=prefill
        )
        x = residual + sa_output

        # Cross-Attention Sub Layer
        residual = x
        x_norm = self.pre_ca_norm(x)

        ca_output, _ = self.cross_attention(
            X_q=x_norm,
            X_kv=encoder_output,
            q_pos=tgt_pos, # Query position from decoder
            kv_pos=src_pos, # Key/Value position from encoder (if RoPE applied to K/V encoder when precompute)
            deterministic=deterministic,
            attn_mask=cross_attn_mask,
            cache=cross_attn_kv_cache,
            prefill=True
        )
        x = residual + ca_output

        # Feed Forward Sub Layer
        residual = x
        x_norm = self.pre_ff_norm(x)
        
        ff_output = self.ff(
            x_norm,
            deterministic=deterministic
        )
        x = residual + ff_output

        return x, new_self_attn_kv_for_cache # Return new K/V only from self-attention



class SabdaModel(nn.Module):
    def __init__(self, config: SabdaConfig):
        super().__init__()
        self.config = config
        model_cfg = config.model
        encoder_cfg = config.model.encoder
        decoder_cfg = config.model.decoder
        data_cfg = config.data

        # Encoder Components
        self.text_embeddings = nn.Embedding(
            model_cfg.src_vocab_size,
            encoder_cfg.d_embd
        )
        self.encoder_dropout = nn.Dropout(model_cfg.dropout)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_embd=encoder_cfg.d_embd,
                    n_heads=encoder_cfg.n_heads,
                    d_heads=encoder_cfg.d_heads,
                    d_ff=encoder_cfg.d_ff,
                    ff_activations=tuple(encoder_cfg.ff_activations),
                    dropout_rate=model_cfg.dropout,
                    norm_eps=model_cfg.norm_eps,
                    weight_dtype=_str_to_dtype(model_cfg.weight_dtype),
                    min_rope_timescale=model_cfg.min_rope_timescale,
                    max_rope_timescale=model_cfg.max_rope_timescale,
                    rope_dtype=_str_to_dtype(config.train_args.dtype)
                ) for _ in range(encoder_cfg.n_layer)
            ]
        )

        self.encoder_norm = RMSNorm(encoder_cfg.d_embd, eps=model_cfg.norm_eps)

        # Decoder Components
        self.audio_embeddings = nn.ModuleList([
            nn.Embedding(model_cfg.tgt_vocab_size, decoder_cfg.d_embd)
            for _ in range(data_cfg.channels)
        ])

        self.decoder_dropout = nn.Dropout(model_cfg.dropout)

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_embd_decoder=decoder_cfg.d_embd,
                    d_embd_encoder=encoder_cfg.d_embd,
                    n_gqa_query_heads=decoder_cfg.n_gqa_heads,
                    n_gqa_kv_heads=decoder_cfg.kv_heads,
                    d_gqa_head=decoder_cfg.d_gqa_heads,
                    n_cross_query_heads=decoder_cfg.n_cross_heads,
                    d_cross_head=decoder_cfg.d_cross_heads,
                    d_ff=decoder_cfg.d_ff,
                    ff_activations=tuple(decoder_cfg.ff_activations),
                    dropout_rate=model_cfg.dropout,
                    norm_eps=model_cfg.norm_eps,
                    weight_dtype=_str_to_dtype(model_cfg.weight_dtype),
                    min_rope_timescale=model_cfg.min_rope_timescale,
                    max_rope_timescale=model_cfg.max_rope_timescale,
                    rope_dtype=_str_to_dtype(config.train_args.dtype)
                ) for _ in range(decoder_cfg.n_layer)
            ]
        )

        self.decoder_norm = RMSNorm(decoder_cfg.d_embd, eps=model_cfg.norm_eps)

        self.logits_projection = DenseGeneral(
            in_shapes=(decoder_cfg.d_embd,),
            # Output: (NumChannelsDAC, TargetVocabSize)
            out_features=(data_cfg.channels, model_cfg.tgt_vocab_size),
            axis=(-1,),
            weight_dtype=_str_to_dtype(model_cfg.weight_dtype)
        )
        self.logits_in_fp32 = config.train_args.logits_dot_in_fp32
    
    def forward(
        self,
        src_tokens: torch.Tensor, # (B, S_txt)
        tgt_tokens: torch.Tensor, # (B, T_tgt, C) - DAC target tokens (already delayed and have BOS/EOS)
        src_pos: torch.Tensor, # (B, S_txt)
        tgt_pos: torch.Tensor, # (B, T_tgt)
        enc_self_attn_mask: torch.Tensor, # (B, 1, S_txt, S_txt)
        dec_self_attn_mask: torch.Tensor, # (B, 1, T_tgt, T_tgt) - causal + padding
        dec_cross_attn_mask: torch.Tensor, # (B, 1, T_tgt, S_txt) - padding
        deterministic: bool = True
    ) -> torch.Tensor: # return logits (B, T_tgt, C, V_tgt)
        
        # Encoder Pass
        encoder_input_emb = self.text_embeddings(src_tokens)
        if not deterministic:
            encoder_input_emb = self.encoder_dropout(encoder_input_emb)

        # Looping through all Encoder Layers
        encoder_hidden_states = encoder_input_emb
        for encoder_layer in self.encoder_layers:
            encoder_hidden_states = encoder_layer(
                x=encoder_hidden_states,
                src_pos=src_pos,
                attn_mask=enc_self_attn_mask,
                deterministic=deterministic
            )

        # Final Encoder Normalization
        encoder_output = self.encoder_norm(encoder_hidden_states)

        # Decoder Pass
        decoder_input_x = None
        for i in range(self.config.data.channels):
            channel_target_tokens = tgt_tokens[..., i] # (B, T_tgt)
            channel_target_emb = self.audio_embeddings[i](channel_target_tokens) # (B, T_tgt, d_embd decoder)
            if decoder_input_x is None:
                decoder_input_x = channel_target_emb
            else:
                decoder_input_x = decoder_input_x + channel_target_emb

        if not deterministic:
            decoder_input_x = self.decoder_dropout(decoder_input_x)

        
        # Prepare KVCache
        device = src_tokens.device 
        B = src_tokens.shape[0]
        T_tgt = tgt_tokens.shape[1] 
        S_src = encoder_output.shape[1] 

        # KVCache for SELF-ATTENTION (filled with prefill=True)
        self_attention_caches_list = [
            KVCache(
                n_heads=self.config.model.decoder.n_gqa_heads, 
                max_len=T_tgt, 
                d_heads=self.config.model.decoder.d_gqa_heads,
                device=device,
                batch_size=B # Gunakan batch size aktual
            ) for _ in range(self.config.model.decoder.n_layer)
        ]

        # Pre-compute KVCache for CROSS-ATTENTION
        cross_attention_caches_list = []

        for i in range(self.config.model.decoder.n_layer):
            decoder_layer_module = self.decoder_layers[i]
            cross_attn_submodule = decoder_layer_module.cross_attention
            
            # Compute Key from encoder_output
            k_cross = cross_attn_submodule.W_K(encoder_output) 
            if src_pos is not None: # Terapkan RoPE pada Key cross-attn jika ada src_pos
                 k_cross = cross_attn_submodule.rope(k_cross, position=src_pos)
            k_cross = k_cross.transpose(1, 2) 

            # Compute Value from encoder output
            v_cross = cross_attn_submodule.W_V(encoder_output)
            v_cross = v_cross.transpose(1, 2) 
            
            cross_cache_instance = KVCache(
                n_heads=cross_attn_submodule.n_query_heads, 
                max_len=S_src, 
                d_heads=cross_attn_submodule.d_heads,
                device=device,
                batch_size=B,
                k=k_cross, 
                v=v_cross  
            )
            cross_cache_instance.current_idx = S_src 
            cross_attention_caches_list.append(cross_cache_instance)

        # Looping through all decoder layers
        decoder_hidden_states = decoder_input_x
        
        for i, decoder_layer in enumerate(self.decoder_layers): # Gunakan enumerate
            decoder_hidden_states, _ = decoder_layer(
                x=decoder_hidden_states,
                encoder_output=encoder_output,
                tgt_pos=tgt_pos,
                src_pos=src_pos,
                self_attn_mask=dec_self_attn_mask,
                cross_attn_mask=dec_cross_attn_mask,
                self_attn_kv_cache=self_attention_caches_list[i],
                cross_attn_kv_cache=cross_attention_caches_list[i],
                deterministic=deterministic,
                prefill=True 
            )

        # Final Decoder Normalization
        decoder_output = self.decoder_norm(decoder_hidden_states)

        # Logits Projection
        logits = self.logits_projection(decoder_output) # (B, T_tgt, C, V_tgt)

        if self.logits_in_fp32 and logits.dtype != torch.float32:
            logits = logits.to(torch.float32)

        return logits
    
    def decode_one_step(
        self,
        current_tgt_tokens: torch.Tensor,    # Shape: (B, 1, C) - Input DAC tokens for the current step (B=batch_size, C=num_channels)
        current_tgt_pos: torch.Tensor,       # Shape: (B, 1) - Current time position
        encoder_output: torch.Tensor,        # Shape: (B, S_txt, D_encoder) - Output from the encoder
        src_pos: Optional[torch.Tensor],     # Shape: (B, S_txt) - Positions for encoder output (for RoPE in cross-attn if needed)
        dec_cross_attn_mask_step: Optional[torch.Tensor], # Shape: (B, N_cross_heads, 1, S_txt) - Cross-attention mask
        self_attention_caches_list: List[KVCache],  # List of KVCache objects for self-attention (updated in-place)
        cross_attention_caches_list: List[KVCache]  # List of pre-computed KVCache objects for cross-attention (read-only)
    ) -> torch.Tensor:                          # Returns logits of shape (B, 1, C, V_tgt)
        
        deterministic = True # Always deterministic for generation steps
        device = current_tgt_tokens.device
        batch_size = current_tgt_tokens.shape[0]
        num_channels = self.config.data.channels

        # 1. Input Embedding for the current step
        # current_tgt_tokens is (B, 1, C)
        decoder_hidden_states_step = None
        for i in range(num_channels):
            channel_token_ids = current_tgt_tokens[:, 0, i]  # (B,)
            channel_emb = self.audio_embeddings[i](channel_token_ids)  # (B, D_decoder)
            if decoder_hidden_states_step is None:
                decoder_hidden_states_step = channel_emb.unsqueeze(1)  # (B, 1, D_decoder)
            else:
                decoder_hidden_states_step = decoder_hidden_states_step + channel_emb.unsqueeze(1)
        
        # No dropout during deterministic generation
        # if not deterministic:
        #     decoder_hidden_states_step = self.decoder_dropout(decoder_hidden_states_step)

        # 2. Loop through Decoder Layers
        for layer_idx, decoder_layer in enumerate(self.decoder_layers):
            self_attn_cache_for_layer = self_attention_caches_list[layer_idx]
            cross_attn_cache_for_layer = cross_attention_caches_list[layer_idx]

            # Output from layer, and the (key, value) of the current step's self-attention for caching
            decoder_hidden_states_step, sa_key_value_current_step = decoder_layer.forward(
                x=decoder_hidden_states_step,              # (B, 1, D_decoder)
                encoder_output=encoder_output,             # (B, S_txt, D_encoder)
                tgt_pos=current_tgt_pos,                   # (B, 1)
                src_pos=src_pos,                           # (B, S_txt)
                self_attn_mask=None,                       # For single step with KVCache, causality is handled by cache
                cross_attn_mask=dec_cross_attn_mask_step,  # (B, N_cross_heads, 1, S_txt)
                self_attn_kv_cache=self_attn_cache_for_layer,
                cross_attn_kv_cache=cross_attn_cache_for_layer,
                deterministic=deterministic,
                prefill=False  # CRITICAL: prefill is False for autoregressive self-attention steps
            )
            
            # Update the self-attention KVCache with the K,V from the current step
            if sa_key_value_current_step is not None:
                # sa_key_value_current_step should be a tuple (k_current_step, v_current_step)
                # each of shape (B, n_gqa_heads, 1, d_gqa_heads)
                self_attn_cache_for_layer.update_cache(sa_key_value_current_step[0], sa_key_value_current_step[1])
            else:
                # This case should ideally not happen if KVCache is active for self-attention
                logger_layers = logging.getLogger(__name__) # Get a logger instance
                logger_layers.warning(f"Self-attention for layer {layer_idx} did not return K/V for cache update during decode_one_step.")


        # 3. Final Normalization and Logits Projection
        decoder_output_final_step = self.decoder_norm(decoder_hidden_states_step)
        logits_step = self.logits_projection(decoder_output_final_step)  # (B, 1, C, V_tgt)

        if self.logits_in_fp32 and logits_step.dtype != torch.float32:
            logits_step = logits_step.to(torch.float32)

        return logits_step