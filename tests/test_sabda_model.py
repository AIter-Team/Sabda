import torch
import torch.nn.functional as F
from sabda.config_schema import SabdaConfig # dan sub-kelas config lainnya
from sabda.utils import _str_to_dtype, _normalize_axes, get_activation_fn # Sesuaikan dengan isi utils Anda
from sabda.layers import (
    DenseGeneral, GatedMLPFeedForward, # atau nama FFN Anda
    RotaryPositionEmbedding, KVCache, Attention, 
    EncoderLayer, DecoderLayer, SabdaModel 
)

def run_initial_test():
    torch.cuda.empty_cache()

    print("Memulai pengujian awal SabdaModel...")
    # Pilih device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # 1. Muat Konfigurasi
    # Ganti dengan path yang benar ke file konfigurasi Anda jika perlu.
    # Jika file ini ada di tests/test_sabda_model.py, dan training_configs/ ada di level yang sama dengan tests/,
    # path relatif dari root proyek mungkin "../training_configs/sabda_v1_config.json"
    # atau Anda bisa menggunakan path absolut.
    # Untuk kemudahan, jika path tidak ditemukan, kita gunakan config default.
    config_path = "E:/Workspace/Audio/Sabda/training_configs/sabda_v1_config.json" # Sesuaikan path ini dari root proyek Anda
    try:
        # Asumsikan SabdaConfig memiliki metode load seperti yang kita diskusikan
        # Jika tidak, Anda bisa menginstansiasi SabdaConfig() secara langsung
        # dengan nilai default jika file JSON belum siap.
        # config = SabdaConfig.load(config_path) # Jika metode load Anda adalah metode kelas
        # Untuk contoh ini, kita buat config default secara manual agar lebih portabel
        # Ini harusnya diisi dengan memuat file JSON Anda.
        config = SabdaConfig.load(config_path) # Menggunakan default dari Pydantic jika file JSON tidak ada
                               # atau ganti dengan pemuatan dari file JSON Anda
        print(f"SabdaConfig berhasil dimuat/dibuat. Menggunakan versi: {config.version}")
        print(f"  Encoder d_embd: {config.model.encoder.d_embd}, n_heads: {config.model.encoder.n_heads}, d_heads: {config.model.encoder.d_heads}")
        # Validasi dimensi encoder
        if config.model.encoder.d_embd != config.model.encoder.n_heads * config.model.encoder.d_heads:
            print(f"Peringatan: Dimensi Encoder mungkin tidak konsisten: d_embd ({config.model.encoder.d_embd}) "
                  f"!= n_heads ({config.model.encoder.n_heads}) * d_heads ({config.model.encoder.d_heads})")

    except FileNotFoundError:
        print(f"File konfigurasi tidak ditemukan di {config_path}. Menggunakan SabdaConfig default.")
        config = SabdaConfig() # Gunakan default jika file tidak ada
    except Exception as e:
        print(f"Error saat memuat/membuat SabdaConfig: {e}")
        print("Pastikan file JSON konfigurasi Anda sudah benar atau definisi Pydantic default sudah tepat.")
        return

    # 2. Inisialisasi SabdaModel
    try:
        print("Init model..")
        model = SabdaModel(config)
        model.to(device) # Pindahkan model ke device yang dipilih
        model.eval() # Set ke mode eval untuk menonaktifkan dropout (jika deterministic tidak diatur di forward)
        print("SabdaModel berhasil diinisialisasi dan dipindahkan ke device.")
    except Exception as e:
        print(f"Error saat inisialisasi SabdaModel: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Buat Tensor Input Dummy
    batch_size = 1 # Contoh batch size
    
    # Encoder Inputs
    src_len = config.data.txt_len 
    src_tokens = torch.randint(0, config.model.src_vocab_size, (batch_size, src_len), device=device)
    src_pos = torch.arange(0, src_len, device=device).unsqueeze(0).expand(batch_size, -1)
    # Mask padding sederhana untuk encoder (asumsi semua non-padding)
    enc_padding_mask_1d = torch.ones(batch_size, src_len, dtype=torch.bool, device=device)
    enc_self_attn_mask = (enc_padding_mask_1d.unsqueeze(2) & enc_padding_mask_1d.unsqueeze(1)).unsqueeze(1)


    # Decoder Inputs
    # Untuk training, tgt_tokens adalah target yang sudah di-shift + BOS.
    # collate_fn akan menangani BOS/EOS dan delay pattern. Untuk tes ini, kita buat sederhana.
    # Panjang target untuk input decoder (misalnya, sudah termasuk BOS dan tanpa EOS akhir)
    tgt_len_input = config.data.audio_len # Panjang sekuens audio yang di-feed ke decoder
    
    tgt_tokens_input = torch.randint(
        0, 
        config.model.tgt_vocab_size, 
        (batch_size, tgt_len_input, config.data.channels), 
        device=device
    )
    tgt_pos_input = torch.arange(0, tgt_len_input, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Causal mask + padding mask untuk decoder self-attention
    dec_padding_mask_1d = torch.ones(batch_size, tgt_len_input, dtype=torch.bool, device=device) # Asumsi non-padding
    causal_mask_2d = torch.tril(torch.ones(tgt_len_input, tgt_len_input, dtype=torch.bool, device=device))
    dec_self_attn_mask = (dec_padding_mask_1d.unsqueeze(2) & dec_padding_mask_1d.unsqueeze(1) & causal_mask_2d).unsqueeze(1)
    
    # Cross-attention mask (padding untuk output encoder)
    dec_cross_attn_mask = (dec_padding_mask_1d.unsqueeze(2) & enc_padding_mask_1d.unsqueeze(1)).unsqueeze(1)
    
    print("\nTensor input dummy berhasil dibuat:")
    print(f"  src_tokens shape: {src_tokens.shape}")
    print(f"  tgt_tokens_input shape: {tgt_tokens_input.shape}") # Ini adalah input ke decoder
    print(f"  src_pos shape: {src_pos.shape}")
    print(f"  tgt_pos_input shape: {tgt_pos_input.shape}")
    print(f"  enc_self_attn_mask shape: {enc_self_attn_mask.shape}")
    print(f"  dec_self_attn_mask shape: {dec_self_attn_mask.shape}")
    print(f"  dec_cross_attn_mask shape: {dec_cross_attn_mask.shape}")

    # 4. Panggil SabdaModel.forward()
    try:
        print("\nMencoba memanggil SabdaModel.forward()...")
        # PENTING: Pastikan nama parameter di SabdaModel.forward sesuai dengan yang Anda definisikan
        logits = model.forward(
            src_tokens=src_tokens,
            tgt_tokens=tgt_tokens_input, # Ini adalah input ke decoder
            src_pos=src_pos,
            tgt_pos=tgt_pos_input, # Posisi untuk input decoder
            enc_self_attn_mask=enc_self_attn_mask,
            dec_self_attn_mask=dec_self_attn_mask,
            dec_cross_attn_mask=dec_cross_attn_mask,
            deterministic=True # Gunakan True untuk tes agar konsisten (menonaktifkan dropout)
        )
        print("SabdaModel.forward() berhasil dipanggil.")
        print(f"  Output logits shape: {logits.shape}")

        # Verifikasi shape output logits
        # Harusnya (batch_size, tgt_len_input, num_channels_dac, tgt_vocab_size)
        expected_logits_shape = (
            batch_size, 
            tgt_len_input, 
            config.data.channels, 
            config.model.tgt_vocab_size
        )
        assert logits.shape == expected_logits_shape, \
            f"Bentuk logits tidak sesuai! Diharapkan {expected_logits_shape}, didapatkan {logits.shape}"
        print("Bentuk output logits sudah sesuai!")

        # 5. (Opsional) Pemeriksaan Gradien
        print("\nMencoba backward pass sederhana...")
        # Buat target dummy untuk loss (dengan shape yang sama dengan logits, tapi isinya token ID)
        dummy_target_for_loss = torch.randint(
            0, config.model.tgt_vocab_size,
            (batch_size, tgt_len_input, config.data.channels),
            device=device, dtype=torch.long
        )
        
        # Hitung loss dummy (misalnya, cross entropy)
        # Logits perlu di-reshape untuk cross_entropy: (N, C_classes, ...)
        # Target juga perlu di-reshape: (N, ...)
        # Kita ambil channel pertama saja untuk contoh loss sederhana
        loss_logits = logits[..., 0, :].contiguous().view(-1, config.model.tgt_vocab_size) # (B*T, V_tgt)
        loss_targets = dummy_target_for_loss[..., 0].contiguous().view(-1) # (B*T)
        
        dummy_loss = F.cross_entropy(loss_logits, loss_targets)
        
        dummy_loss.backward()
        print(f"Backward pass berhasil. Loss dummy: {dummy_loss.item()}")
        
        # Periksa apakah beberapa parameter memiliki gradien
        has_grads = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                # print(f"  Parameter {name} memiliki gradien (contoh: {param.grad.abs().mean().item()}).")
                has_grads = True
                break 
        if has_grads:
            print("Beberapa parameter memiliki gradien.")
        else:
            print("Peringatan: Tidak ada parameter yang memiliki gradien setelah backward pass.")
        
        model.zero_grad() # Bersihkan gradien setelah tes

    except Exception as e:
        print(f"Error saat memanggil SabdaModel.forward() atau backward(): {e}")
        import traceback
        traceback.print_exc()

# Panggil fungsi pengujian jika skrip ini dijalankan secara langsung
if __name__ == '__main__':
    run_initial_test()