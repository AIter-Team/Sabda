# SABDA/tests/test_dataloader_real_data.py

import torch
from torch.utils.data import DataLoader
import os
import pandas as pd # Diimpor jika Anda ingin membuat metadata dummy secara terprogram
import dac # Pastikan library dac sudah terinstal dan bisa diimpor
from typing import Dict, List, Tuple, Optional # Untuk type hints

# --- Impor dari paket sabda_tts Anda ---
# Sesuaikan path impor ini dengan struktur proyek Anda
# Ini mengasumsikan skrip dijalankan dari root SABDA/ menggunakan `python -m tests.test_dataloader_real_data`
from sabda.config_schema import SabdaConfig, DataConfig # dan sub-kelas config lainnya jika dibutuhkan langsung
from sabda.dataloader import SabdaDataset, create_sabda_collate_fn # Impor kelas dan fungsi Anda
from sabda.audio_processing import build_delay_indices, apply_audio_delay # Impor jika belum ada di dataloader.py

# (Jika ada fungsi utilitas lain yang dibutuhkan oleh collate_fn secara tidak langsung,
#  pastikan mereka juga bisa diakses/diimpor dengan benar oleh modul dataloader Anda)


def test_sabda_dataloader_with_real_data(
    dataset_base_name: str, # Nama direktori dataset Anda di dalam SABDA/datasets/
    config_filename: str    # Nama file konfigurasi Anda di dalam SABDA/training_configs/
):
    print("Memulai pengujian SabdaDataset dan DataLoader dengan dataset nyata...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # 1. Tentukan Path Dataset dan Konfigurasi
    # Dapatkan path direktori root proyek saat ini (tempat SABDA/ berada)
    # Ini mengasumsikan skrip tes ada di SABDA/tests/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.abspath(os.path.join(current_script_dir, "..")) # Naik satu level ke SABDA/

    dataset_base_path = os.path.join(project_root_dir, "datasets", dataset_base_name)
    config_json_path = os.path.join(project_root_dir, "training_configs", config_filename)

    metadata_path = os.path.join(dataset_base_path, "metadata.csv")
    audio_dir = os.path.join(dataset_base_path, "wavs")

    if not os.path.exists(metadata_path):
        print(f"Error: File metadata tidak ditemukan di {metadata_path}")
        return
    if not os.path.exists(audio_dir):
        print(f"Error: Direktori audio tidak ditemukan di {audio_dir}")
        return
        
    print(f"Menggunakan metadata dari: {metadata_path}")
    print(f"Menggunakan audio dari direktori: {audio_dir}")
    print(f"Menggunakan konfigurasi dari: {config_json_path}")

    # 2. Muat Konfigurasi SabdaConfig (dan ekstrak SabdaDataConfig)
    try:
        if not os.path.exists(config_json_path):
            raise FileNotFoundError(f"File konfigurasi SabdaConfig tidak ditemukan di {config_json_path}")
        
        # Asumsikan SabdaConfig memiliki metode load dari JSON (seperti di DiaConfig)
        # Jika tidak, Anda bisa memuat JSON secara manual:
        # import json
        # with open(config_json_path, 'r') as f:
        #     json_data = json.load(f)
        # config_full = SabdaConfig(**json_data)
        
        # Menggunakan metode load yang kita diskusikan untuk SabdaConfig
        config_full = SabdaConfig.load(config_json_path) 
        if config_full is None: # Jika load mengembalikan None karena FileNotFoundError internal
             raise FileNotFoundError(f"Gagal memuat SabdaConfig dari {config_json_path} (load mengembalikan None).")

        data_cfg = config_full.data 
        print(f"SabdaConfig berhasil dimuat. DataConfig txt_len: {data_cfg.txt_len}, audio_len: {data_cfg.audio_len}")
        print(f"  Delay pattern: {data_cfg.delay_pattern}")
        print(f"  Audio PAD: {data_cfg.audio_pad_value}, BOS: {data_cfg.audio_bos_value}, EOS: {data_cfg.audio_eos_value}")

    except Exception as e:
        print(f"Error saat memuat/membuat SabdaConfig atau SabdaDataConfig: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Inisialisasi Model DAC
    try:
        print("Memuat model DAC...")
        dac_model = dac.DAC.load(dac.utils.download())
        dac_model.to(device)
        dac_model.eval() 
        print("Model DAC berhasil dimuat.")
    except Exception as e:
        print(f"Error memuat model DAC: {e}")
        return

    # 4. Inisialisasi SabdaDataset
    try:
        sabda_dataset = SabdaDataset(
            dataset_base_path=dataset_base_path, # Mengoper dataset_base_path
            data_config=data_cfg,
            dac_model=dac_model,
            target_sample_rate=44100, # Sesuaikan jika DAC Anda atau data_config Anda memiliki nilai berbeda
            # max_samples=5, # Batasi jumlah sampel untuk tes awal agar cepat
            metadata_filename="metadata.csv", # Nama file metadata di dalam dataset_base_path
            audio_subdir="wavs" # Nama subdirektori audio di dalam dataset_base_path
        )
        print(f"SabdaDataset berhasil diinisialisasi dengan {len(sabda_dataset)} sampel (dibatasi oleh max_samples).")
        
        if len(sabda_dataset) == 0:
            print("Tidak ada sampel yang dimuat oleh SabdaDataset. Periksa metadata dan path.")
            return

        print("\nMengambil sampel pertama dari SabdaDataset...")
        raw_text_sample, dac_tokens_sample = sabda_dataset[0]
        print(f"  Teks Mentah Sampel[0]: '{raw_text_sample}'")
        print(f"  Token DAC Sampel[0] Shape: {dac_tokens_sample.shape}") 
        print(f"  Token DAC Sampel[0] Dtype: {dac_tokens_sample.dtype}")
        assert dac_tokens_sample.ndim == 2 and dac_tokens_sample.shape[1] == data_cfg.channels, \
            "Shape token DAC dari dataset tidak sesuai!"
        print("Pengambilan sampel pertama dari SabdaDataset berhasil.")

    except Exception as e:
        print(f"Error saat inisialisasi atau __getitem__ SabdaDataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Buat dan Inisialisasi DataLoader
    try:
        collate_fn_instance = create_sabda_collate_fn(data_config=data_cfg, device=device)
        
        test_batch_size = 1 
        if len(sabda_dataset) == 0:
            print("Dataset kosong, tidak bisa membuat DataLoader.")
            return
        if len(sabda_dataset) < test_batch_size:
            print(f"Jumlah sampel ({len(sabda_dataset)}) lebih kecil dari batch_size ({test_batch_size}), menggunakan batch_size = {len(sabda_dataset)}")
            test_batch_size = len(sabda_dataset)

        dataloader = DataLoader(
            sabda_dataset,
            batch_size=test_batch_size,
            collate_fn=collate_fn_instance,
            shuffle=False 
        )
        print("\nDataLoader berhasil dibuat.")
    except Exception as e:
        print(f"Error saat membuat DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Iterasi melalui DataLoader dan Periksa Batch
    try:
        print("\nMengiterasi melalui DataLoader...")
        batch_count = 0
        for i, batch in enumerate(dataloader):
            batch_count +=1
            print(f"\n--- Batch {i+1} ---")
            expected_keys = [
                'src_tokens', 'src_positions', 'enc_self_attn_mask',
                'tgt_tokens', 'tgt_positions', 'dec_self_attn_mask', 
                'dec_cross_attn_mask', 'tgt_lens'
            ]
            for key in expected_keys:
                assert key in batch, f"Kunci '{key}' tidak ditemukan dalam batch!"
                tensor = batch[key]
                print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")

            B = batch['src_tokens'].shape[0]
            assert batch['src_tokens'].shape == (B, data_cfg.txt_len)
            
            expected_tgt_len_output = data_cfg.audio_len + 2
            assert batch['tgt_tokens'].shape == (B, expected_tgt_len_output, data_cfg.channels)
            assert batch['tgt_lens'].shape == (B,)
            if B > 0 : # Hanya periksa jika batch tidak kosong
                 assert batch['tgt_lens'].max().item() <= expected_tgt_len_output, \
                    f"tgt_lens ({batch['tgt_lens'].max().item()}) melebihi expected_tgt_len_output ({expected_tgt_len_output})"
                 print(f"  Contoh src_tokens[0, :10]: {batch['src_tokens'][0, :10]}")
                 print(f"  Contoh tgt_tokens[0, :5, 0]: {batch['tgt_tokens'][0, :5, 0]}")
                 print(f"  tgt_lens[0]: {batch['tgt_lens'][0]}")

            if i >= 1: # Uji hanya 2 batch untuk awal
                break
        
        if batch_count > 0:
            print(f"\nPengujian {batch_count} batch dari DataLoader berhasil!")
            print("\nPengujian DataLoader pipeline dengan data nyata selesai dengan sukses!")
        else:
            print("\nTidak ada batch yang diproses dari DataLoader. Periksa dataset Anda.")


    except Exception as e:
        print(f"Error saat iterasi melalui DataLoader atau memeriksa batch: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # --- PENTING: SESUAIKAN PATH INI ---
    # Ganti "nama_dataset_anda" dengan nama direktori dataset Anda yang sebenarnya
    # yang ada di dalam SABDA/datasets/
    nama_direktori_dataset_anda = "E:\Workspace\Audio\Sabda\datasets\RP_Short" # <--- GANTI INI
    
    # Nama file konfigurasi JSON Anda di dalam SABDA/training_configs/
    nama_file_config_anda = "sabda_v1_config.json" # <--- GANTI INI JIKA PERLU

    print("Peringatan: Skrip ini mengasumsikan semua definisi kelas SabdaTTS Anda")
    print(" (SabdaConfig, SabdaDataset, create_sabda_collate_fn, dll.) sudah tersedia")
    print(" dan bisa diimpor dengan benar dari paket sabda_tts.")
    print(f"Pastikan nama direktori dataset ('{nama_direktori_dataset_anda}') dan ")
    print(f"nama file config ('{nama_file_config_anda}') sudah benar di dalam skrip ini.")
    
    test_sabda_dataloader_with_real_data(
        dataset_base_name=nama_direktori_dataset_anda,
        config_filename=nama_file_config_anda
    )