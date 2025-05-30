import torch
from typing import List, Tuple

# Impor SabdaDataConfig jika delay_pattern, pad_value, bos_value akan diambil dari sana
# atau terima sebagai argumen fungsi.
# from .config_schema import SabdaDataConfig 

def build_delay_indices(
    B: int, 
    T: int, 
    C: int, 
    delay_pattern: List[int], 
    device: torch.device # Tambahkan device untuk pembuatan tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute (t_idx_BxTxC, indices_BTCx3) so that out[t, c] = in[t - delay[c], c].
    Negative t_idx => BOS; t_idx >= T => PAD.
    (Diadaptasi dari dia/audio.py)
    """
    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32, device=device) # Buat di device yang benar

    # t_idx_BxT: Indeks waktu asli untuk setiap sampel dalam batch
    # Shape: [B, T]
    t_idx_BxT = torch.broadcast_to(
        torch.arange(T, dtype=torch.int32, device=device)[None, :],
        [B, T],
    )
    
    # t_idx_BxTx1: Tambah dimensi channel untuk broadcasting dengan delay_arr
    # Shape: [B, T, 1]
    t_idx_BxTx1 = t_idx_BxT.unsqueeze(-1)
    
    # t_idx_BxTxC: Indeks waktu target setelah delay
    # delay_arr.view(1, 1, C) agar bisa di-broadcast
    # Shape: [B, T, C]
    t_idx_BxTxC = t_idx_BxTx1 - delay_arr.view(1, 1, C)

    # b_idx_BxTxC: Indeks batch untuk setiap posisi
    # Shape: [B, T, C]
    b_idx_BxTxC = torch.broadcast_to(
        torch.arange(B, dtype=torch.int32, device=device).view(B, 1, 1),
        [B, T, C],
    )
    
    # c_idx_BxTxC: Indeks channel untuk setiap posisi
    # Shape: [B, T, C]
    c_idx_BxTxC = torch.broadcast_to(
        torch.arange(C, dtype=torch.int32, device=device).view(1, 1, C),
        [B, T, C],
    )

    # t_clamped_BxTxC: Indeks waktu yang sudah di-clamp agar valid untuk gather
    # Shape: [B, T, C]
    t_clamped_BxTxC = torch.clamp(t_idx_BxTxC, 0, T - 1)

    # indices_BTCx3: Indeks untuk operasi gather
    # Shape: [B*T*C, 3]
    indices_BTCx3 = torch.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_clamped_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        dim=1,
    ).long()  # Pastikan tipe Long untuk indexing

    return t_idx_BxTxC, indices_BTCx3


def apply_audio_delay(
    audio_BxTxC: torch.Tensor, # Shape: (B, T, C)
    pad_value: int,
    bos_value: int,
    precomputed_indices: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Applies the delay pattern to batched audio tokens using precomputed indices,
    inserting BOS where t_idx < 0 and PAD where t_idx >= T.
    (Diadaptasi dari dia/audio.py)
    """
    t_idx_BxTxC, indices_BTCx3 = precomputed_indices
    
    # Pindahkan indeks ke device yang sama dengan audio_BxTxC jika belum
    # (Meskipun build_delay_indices sudah menerima device, ini untuk keamanan)
    t_idx_BxTxC = t_idx_BxTxC.to(audio_BxTxC.device)
    indices_BTCx3 = indices_BTCx3.to(audio_BxTxC.device)

    # Operasi gather menggunakan advanced indexing PyTorch
    # audio_BxTxC[batch_indices, time_indices, channel_indices]
    # indices_BTCx3[:, 0] adalah batch_indices yang sudah di-flatten
    # indices_BTCx3[:, 1] adalah time_indices_clamped yang sudah di-flatten
    # indices_BTCx3[:, 2] adalah channel_indices yang sudah di-flatten
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.shape) # Kembalikan ke shape asli [B, T, C]

    # Buat mask untuk BOS dan PAD
    # Semua tensor harus berada di device yang sama
    mask_bos = t_idx_BxTxC < 0  # True di mana t_idx target < 0
    mask_pad = t_idx_BxTxC >= audio_BxTxC.shape[1]  # True di mana t_idx target >= T asli

    # Buat tensor skalar untuk BOS dan PAD di device yang benar
    bos_tensor = torch.tensor(bos_value, dtype=audio_BxTxC.dtype, device=audio_BxTxC.device)
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=audio_BxTxC.device)

    # Terapkan mask:
    # Jika mask_bos, isi dengan bos_tensor
    # Jika tidak, cek mask_pad: jika mask_pad, isi dengan pad_tensor
    # Jika tidak keduanya, gunakan nilai dari gathered_BxTxC
    result_BxTxC = torch.where(
        mask_bos, 
        bos_tensor, 
        torch.where(mask_pad, pad_tensor, gathered_BxTxC)
    )

    return result_BxTxC


def build_revert_indices(
    B: int, 
    T: int, 
    C: int, 
    delay_pattern: List[int],
    device: torch.device  # <<< ADD device parameter here
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute indices for the revert operation using PyTorch.
    (Adapted from Dia's audio.py and modified to accept device)
    """
    # Use the passed 'device' argument for tensor creation
    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32, device=device)

    # t_idx_BT1: Indeks waktu asli untuk setiap sampel dalam batch, unsqueezed untuk channel
    t_idx_BT1 = torch.arange(T, device=device).unsqueeze(0).expand(B, -1) # Shape [B, T]
    t_idx_BT1 = t_idx_BT1.unsqueeze(-1) # Shape [B, T, 1] for broadcasting with delay_arr

    # t_idx_BxTxC: Indeks waktu target setelah "un-delaying"
    # This computes t' = t + delay[c], then clamps it to be within bounds [0, T-1]
    # Dia's logic for revert is effectively t_target = min(t_current + delay[c], T_max_original -1)
    t_idx_target_BxTxC = torch.minimum(
        t_idx_BT1 + delay_arr.view(1, 1, C),
        torch.tensor(T - 1, dtype=torch.int32, device=device) # Clamp to max valid index T-1
    )

    # b_idx_BxTxC: Indeks batch untuk setiap posisi
    b_idx_BxTxC = torch.arange(B, device=device).view(B, 1, 1).expand(B, T, C)
    
    # c_idx_BxTxC: Indeks channel untuk setiap posisi
    c_idx_BxTxC = torch.arange(C, device=device).view(1, 1, C).expand(B, T, C)

    # indices_BTCx3: Indeks untuk operasi gather
    # We are gathering from the *delayed* input using these *target* indices.
    # The values at t_idx_target_BxTxC in the original (non-delayed) sequence
    # correspond to the values at t_idx_BT1 in the delayed sequence.
    # So, indices_BTCx3 should use t_idx_target_BxTxC for the time dimension.
    indices_BTCx3 = torch.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_idx_target_BxTxC.reshape(-1), # Use the calculated target time indices
            c_idx_BxTxC.reshape(-1),
        ],
        dim=1, # Corrected from axis=1 for PyTorch
    ).long()  # Ensure indices are long type

    # The first returned tensor in Dia's audio.py for revert_audio_delay's precomp
    # is `t_idx_BxTxC` which seems to be `t_idx_target_BxTxC` in my naming here.
    # This tensor is used in `revert_audio_delay` to create a mask: `t_idx_BxTxC >= T_tensor`.
    # Let's return `t_idx_target_BxTxC` as the first element to match that usage pattern.
    return t_idx_target_BxTxC, indices_BTCx3

# Ensure your revert_audio_delay also correctly uses the device of audio_BxTxC for new tensors
# (like pad_tensor, T_tensor) if they aren't already.
# Your existing apply_audio_delay seems to do this well.

def revert_audio_delay(
    audio_BxTxC: torch.Tensor, # Delayed audio
    pad_value: int,
    precomputed_indices: Tuple[torch.Tensor, torch.Tensor],
    T_original: int, # Original length T used when building revert indices
) -> torch.Tensor:
    """
    Reverts a delay pattern from batched audio tokens using precomputed indices.
    (Adapted from Dia's audio.py)
    """
    t_idx_target_BxTxC, indices_BTCx3 = precomputed_indices
    device = audio_BxTxC.device 

    t_idx_target_BxTxC = t_idx_target_BxTxC.to(device)
    indices_BTCx3 = indices_BTCx3.to(device)

    # Gather from the input 'audio_BxTxC' (which is the delayed sequence)
    # using the precomputed 'indices_BTCx3'.
    # These indices tell us, for each position in the output "reverted" tensor,
    # where to pick the value from in the input "delayed" tensor.
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.shape) # Reshape to (B, T, C)

    # Create pad_tensor and T_original_tensor on the correct device
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)
    T_original_tensor = torch.tensor(T_original, dtype=torch.int32, device=device) # Compare t_idx with T_original

    # The t_idx_target_BxTxC contains the effective time index in the *original, non-delayed* timeline
    # that each element of the output *reverted* tensor corresponds to.
    # If this effective time index is >= T_original, it means this position in the
    # output tensor corresponds to a time that was beyond the original sequence length
    # (due to un-delaying), so it should be padding.
    # This logic matches Dia's: result_BxTxC = torch.where(t_idx_BxTxC >= T_tensor, pad_tensor, gathered_BxTxC)
    result_BxTxC = torch.where(
        t_idx_target_BxTxC >= T_original_tensor, # Condition for padding
        pad_tensor, 
        gathered_BxTxC
    )
    return result_BxTxC