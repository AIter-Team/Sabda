import os
import pandas as pd
import torch
import torchaudio
from torch.nn.functional import pad
from torch.utils.data import Dataset
from typing import Tuple, List, Optional, Dict


from .config_schema import DataConfig
from .audio_processing import build_delay_indices, apply_audio_delay


class SabdaDataset(Dataset):
    def __init__(
        self,
        dataset_base_path: str,
        data_config: DataConfig, 
        dac_model,
        target_sample_rate: int = 44100, 
        max_samples: Optional[int] = None,
        metadata_filename: str = "metadata.csv",
        audio_subdir: str = "wavs"
    ) -> None:
        super().__init__()

        self.dataset_base_path = dataset_base_path
        self.audio_dir = os.path.join(dataset_base_path, audio_subdir)
        self.metadata_path = os.path.join(dataset_base_path, metadata_filename)
        
        self.data_config = data_config
        self.dac_model = dac_model
        self.target_sample_rate = target_sample_rate

        # Loading metadata file
        try:
            self.metadata_df = pd.read_csv(
                self.metadata_path, 
                sep='|', 
                header=None, 
                names=['audio_filename', 'text'],
                engine='python',
                skipinitialspace=True,
                encoding='utf-8'
            )

        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata {self.metadata_path} not found!")
        
        except Exception as e:
            raise ValueError(f"Failed to parse or load metadata file: {self.metadata_path}. Error: {e}")
        
        if max_samples is not None and max_samples > 0 and max_samples < len(self.metadata_df):
            self.metadata_df = self.metadata_df.iloc[:max_samples]
        
        if len(self.metadata_df) == 0:
            raise ValueError(f"No samples loaded on: {self.metadata_path}. Makesure the file is not empty and have desired format.")

        print(f"Dataset: Successfully loaded {len(self.metadata_df)} sample from {self.metadata_path}")

    def __len__(self) -> int:
        return len(self.metadata_df)
    
    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        if not (0 <= idx < len(self.metadata_df)):
            raise IndexError(f"Indeks {idx} di luar jangkauan untuk metadata dengan panjang {len(self.metadata_df)}")
        
        try:
            metadata_row = self.metadata_df.iloc[idx]
            audio_filename = metadata_row['audio_filename']
            raw_text = metadata_row['text']

        except KeyError as e:
            raise KeyError(f"Kolom tidak ditemukan di Metadata (pastikan ada 'audio_filename' dan 'text'): {e}")
        
        audio_path = os.path.join(self.audio_dir, audio_filename)

        try:
            waveform, original_sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error saat memuat file audio: {audio_path}. Error: {e}")
            raise RuntimeError(f"Gagal memuat audio: {audio_path}") from e
        
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform[0, :].unsqueeze(0)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if original_sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=original_sample_rate,
                new_freq=self.target_sample_rate
            )

        if waveform.ndim == 2: 
            waveform = waveform.unsqueeze(1) 

        waveform = waveform.to(next(self.dac_model.parameters()).device)

        with torch.no_grad():
            try:
                audio_tensor_dac_input = self.dac_model.preprocess(waveform, self.target_sample_rate)
                _, dac_tokens_encoded, *_ = self.dac_model.encode(audio_tensor_dac_input)
                dac_tokens_tensor = dac_tokens_encoded.squeeze(0).transpose(0, 1)
            except Exception as e:
                print(f"Error saat memproses audio {audio_filename} menggunakan DAC: {e}")
                raise RuntimeError(f"Gagal memproses audio {audio_filename} menggunakan DAC") from e
        
        return raw_text, dac_tokens_tensor
         

def create_sabda_collate_fn(data_config: DataConfig) -> callable:

    def sabda_collate_fn(batch: List[Tuple[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # batch adalah list dari tuple, setiap tuple: (raw_text, dac_tokens_tensor)
        # dac_tokens_tensor memiliki shape (num_audio_frames, num_channels_dac)

        texts, dac_tokens_list = zip(*batch)
        current_batch_size = len(texts)

        # === 1. Pemrosesan Teks (Input Encoder) ===
        max_text_len = data_config.txt_len
        text_pad_value = data_config.text_pad_value
        processed_text_ids_list = []
        for text_str in texts:
            byte_text = text_str.encode('utf-8')
            if len(byte_text) > max_text_len:
                byte_text_processed = byte_text[:max_text_len]
            else:
                byte_text_processed = byte_text
            token_ids = list(byte_text_processed)
            padding_needed = max_text_len - len(token_ids)
            if padding_needed > 0:
                token_ids.extend([text_pad_value] * padding_needed)
            processed_text_ids_list.append(torch.tensor(token_ids, dtype=torch.long))
        
        src_tokens = torch.stack(processed_text_ids_list)
        src_positions = torch.arange(max_text_len).unsqueeze(0).expand(current_batch_size, -1).clone()
        src_padding_mask_1d = src_tokens.ne(text_pad_value)
        enc_self_attn_mask = (src_padding_mask_1d.unsqueeze(2) & src_padding_mask_1d.unsqueeze(1)).unsqueeze(1).clone()

        # === 2. Pemrosesan Token DAC (Input/Target Decoder) ===
        
        # a. Padding/Truncation Awal dac_tokens_list untuk mendapatkan panjang seragam dalam batch (T_audio_content)
        #    dac_tokens_list berisi tensor dengan shape (num_audio_frames, num_channels_dac)
        
        # Tentukan panjang maksimum frame audio dalam batch saat ini
        current_max_frames_in_batch = 0
        for tokens in dac_tokens_list:
            current_max_frames_in_batch = max(current_max_frames_in_batch, tokens.shape[0])
        
        # Batasi panjang maksimum ini dengan audio_len dari config.
        # Ini adalah panjang konten audio SEBELUM delay dan SEBELUM BOS/EOS.
        # Di finetune.py Dia, mereka langsung membatasi dengan config.data.audio_length saat mengambil seq_lens.
        # Kita lakukan hal serupa: panjang konten efektif adalah min(panjang_asli, config.data.audio_len).
        # Kemudian T_audio_content_batch adalah max dari panjang efektif ini dalam satu batch.
        
        actual_content_lengths = [] # Panjang konten efektif untuk setiap sampel (setelah dibatasi audio_len)
        for tokens in dac_tokens_list:
            actual_content_lengths.append(min(tokens.shape[0], data_config.audio_len))
            
        T_audio_content_batch = max(actual_content_lengths) # Panjang konten terpanjang dalam batch (sudah dibatasi audio_len)

        padded_raw_dac_codes_list = []
        for i, tokens in enumerate(dac_tokens_list):
            # Ambil hanya sepanjang actual_content_lengths[i] karena itu yang akan diproses lebih lanjut
            tokens_to_process = tokens[:actual_content_lengths[i], :]
            
            # Pad ke T_audio_content_batch jika lebih pendek
            padding_needed_audio = T_audio_content_batch - tokens_to_process.shape[0]
            if padding_needed_audio > 0:
                # padding tuple: (pad_left_dim_N, pad_right_dim_N, ..., pad_left_dim_0, pad_right_dim_0)
                # Untuk tensor (L, C), kita ingin padding di akhir L (dimensi 0)
                # Jadi, (0,0) untuk dimensi C (dim=1), dan (0, padding_needed_audio) untuk dimensi L (dim=0)
                padded_codes = pad(tokens_to_process, (0, 0, 0, padding_needed_audio), value=data_config.audio_pad_value)
            else:
                padded_codes = tokens_to_process # Sudah T_audio_content_batch atau lebih panjang (sudah dipotong di atas)
            padded_raw_dac_codes_list.append(padded_codes)
            
        # Stack menjadi batch tensor mentah sebelum delay: (B, T_audio_content_batch, C_dac)
        raw_codes_BTC = torch.stack(padded_raw_dac_codes_list)
        
        B, T_curr_content, C_dac = raw_codes_BTC.shape # T_curr_content = T_audio_content_batch
        
        # b. Terapkan delay pattern
        # Fungsi build_delay_indices dan apply_audio_delay dari sabda_tts.audio_processing
        t_idx_map, gather_map = build_delay_indices(
            B, T_curr_content, C_dac, data_config.delay_pattern, device=raw_codes_BTC.device
        )
        delayed_codes_BTC = apply_audio_delay(
            raw_codes_BTC,
            pad_value=data_config.audio_pad_value,
            bos_value=data_config.audio_bos_value, 
            precomputed_indices=(t_idx_map, gather_map)
        )
        # delayed_codes_BTC sekarang memiliki shape (B, T_audio_content_batch, C_dac)
        # dan sudah berisi token BOS di awal sesuai delay.
        # Kontennya efektif telah dipotong/dibatasi oleh data_config.audio_len karena actual_content_lengths
        # dan T_audio_content_batch sudah mempertimbangkan data_config.audio_len.

        # c. Persiapan tgt_tokens akhir dengan BOS dan EOS
        # Panjang target akhir untuk input decoder dan loss adalah data_config.audio_len + 2 (BOS + konten + EOS)
        # Ini adalah panjang tetap untuk semua sampel dalam batch.
        max_tgt_output_len = data_config.audio_len + 2 

        tgt_tokens = torch.full(
            (B, max_tgt_output_len, C_dac), 
            fill_value=data_config.audio_pad_value, 
            dtype=torch.long,
        )
        
        # Tambahkan token BOS di awal setiap sekuens target
        tgt_tokens[:, 0, :] = data_config.audio_bos_value
        
        tgt_final_lengths = [] # Panjang akhir setiap sekuens target termasuk BOS & EOS & padding hingga max_tgt_output_len
        
        for i in range(B):
            # L_content adalah panjang konten audio sebenarnya untuk sampel ini (dari actual_content_lengths[i])
            L_content = actual_content_lengths[i] 
            
            # Salin token DAC yang sudah di-delay (hingga L_content) ke posisi setelah BOS
            # delayed_codes_BTC[i, :L_content, :] memiliki token yang relevan.
            # Kita tempatkan di tgt_tokens[i, 1 : 1 + L_content, :]
            # Pastikan L_content tidak melebihi data_config.audio_len (seharusnya sudah karena T_audio_content_batch)
            # Panjang efektif yang bisa disalin adalah min(L_content, data_config.audio_len)
            # tapi karena T_audio_content_batch sudah dibatasi data_config.audio_len, L_content juga.
            
            # Kita mengambil dari delayed_codes_BTC, yang panjangnya T_audio_content_batch.
            # Kita hanya perlu menyalin sebanyak L_content (panjang asli sampel i, capped by audio_len).
            # Dan kita tempatkan di tgt_tokens, yang memiliki ruang untuk data_config.audio_len konten.
            
            # Panjang konten yang akan disalin ke tgt_tokens (setelah BOS)
            # Tidak boleh lebih dari data_config.audio_len (ruang yang tersedia di tgt_tokens)
            # dan tidak boleh lebih dari L_content (panjang aktual data sampel i dari delayed_codes_BTC)
            num_frames_to_copy = min(L_content, data_config.audio_len)

            tgt_tokens[i, 1 : 1 + num_frames_to_copy, :] = delayed_codes_BTC[i, :num_frames_to_copy, :]
            
            # Tambahkan token EOS setelah konten audio
            eos_position = 1 + num_frames_to_copy
            if eos_position < max_tgt_output_len: # Pastikan EOS muat
                tgt_tokens[i, eos_position, :] = data_config.audio_eos_value
                current_final_len = eos_position + 1 # BOS + Konten + EOS
            else: 
                # Jika num_frames_to_copy == data_config.audio_len, maka eos_position = 1 + data_config.audio_len
                # Ini adalah slot terakhir di tgt_tokens (indeks max_tgt_output_len - 1).
                # Jadi, ini seharusnya selalu muat jika data_config.audio_len > 0.
                # Jika data_config.audio_len = 0, maka num_frames_to_copy = 0, eos_position = 1.
                # Ini adalah kasus yang aman.
                # Skenario di mana EOS tidak muat hanya jika num_frames_to_copy = data_config.audio_len + 1, yang tidak mungkin.
                # Jadi, warning di implementasi sebelumnya mungkin tidak perlu.
                current_final_len = max_tgt_output_len # Jika EOS ditaruh di slot terakhir, panjangnya max_tgt_output_len

            tgt_final_lengths.append(current_final_len)

        # d. Buat tgt_positions
        tgt_positions = torch.arange(max_tgt_output_len).unsqueeze(0).expand(B, -1).clone()
        
        # e. Buat Mask untuk Decoder
        # Mask padding untuk target decoder (True jika ada channel non-padding)
        tgt_padding_mask_1d = tgt_tokens.ne(data_config.audio_pad_value).any(dim=-1) # (B, max_tgt_output_len)

        # Mask self-attention kausal untuk decoder
        causal_mask_2d = torch.tril(torch.ones((max_tgt_output_len, max_tgt_output_len), dtype=torch.bool))
        dec_self_attn_mask = (tgt_padding_mask_1d.unsqueeze(2) & tgt_padding_mask_1d.unsqueeze(1) & causal_mask_2d).unsqueeze(1).clone()
        
        # Mask cross-attention antara target decoder dan source encoder
        dec_cross_attn_mask = (tgt_padding_mask_1d.unsqueeze(2) & src_padding_mask_1d.unsqueeze(1)).unsqueeze(1).clone()
        
        return {
            'src_tokens': src_tokens,
            'src_positions': src_positions,
            'enc_self_attn_mask': enc_self_attn_mask,
            'tgt_tokens': tgt_tokens, 
            'tgt_positions': tgt_positions,
            'dec_self_attn_mask': dec_self_attn_mask,
            'dec_cross_attn_mask': dec_cross_attn_mask,
            'tgt_lens': torch.tensor(tgt_final_lengths, dtype=torch.long),
        }

    return sabda_collate_fn