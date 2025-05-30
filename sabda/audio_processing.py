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