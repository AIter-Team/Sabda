import torch
import dac # For dac.DAC
from typing import Optional, Dict, List, Tuple # Ensure List and Tuple are imported

# Assuming your project structure allows these imports
from .config_schema import SabdaConfig #
from .layers import SabdaModel, KVCache # KVCache for type hints if we pass them around
from .audio_processing import revert_audio_delay, build_revert_indices

import logging
logger_synthesizer = logging.getLogger(__name__) # For logging within this class

class SabdaSynthesizer:
    def __init__(self, config: SabdaConfig, sabda_model_path: Optional[str] = None, device: Optional[torch.device] = None):
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger_synthesizer.info(f"SabdaSynthesizer will use device: {self.device}")

        # 1. Instantiate SabdaModel
        self.model = SabdaModel(self.config)
        if sabda_model_path:
            try:
                # Load pre-trained SabdaModel weights if path is provided
                self.model.load_state_dict(torch.load(sabda_model_path, map_location=self.device))
                logger_synthesizer.info(f"SabdaModel weights loaded from {sabda_model_path}")
            except Exception as e:
                logger_synthesizer.error(f"Failed to load SabdaModel weights from {sabda_model_path}: {e}", exc_info=True)
                # Decide if you want to raise an error or continue with an uninitialized model
                # For now, let's continue, but log a clear warning.
                logger_synthesizer.warning("Proceeding with a randomly initialized SabdaModel.")

        self.model.to(self.device)
        self.model.eval() # Set to eval mode by default for synthesis
        logger_synthesizer.info(f"SabdaModel initialized/loaded on {self.device} and set to eval mode.")

        # 2. Load DAC Model
        try:
            # dac.utils.download() fetches the default pre-trained model
            self.dac_model = dac.DAC.load(dac.utils.download()) 
            self.dac_model.to(self.device)
            self.dac_model.eval()
            logger_synthesizer.info(f"DAC model loaded on {self.device} and set to eval mode.")
        except Exception as e:
            logger_synthesizer.error(f"Failed to load DAC model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load DAC model: {e}") from e

    # --- Main Generation Method ---
    @torch.inference_mode()
    def generate(
        self,
        text: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: Optional[float] = 0.9, 
        cfg_scale: float = 3.0,
        # TODO: Add audio_prompt_path: Optional[str] = None later
    ) -> torch.Tensor: # Returns audio waveform tensor (e.g., shape [num_samples])
        
        self.model.eval() 
        self.dac_model.eval()

        if max_new_tokens is None:
            max_new_tokens = self.config.data.audio_len # Default to audio_len from config
        if max_new_tokens == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)

        logger_synthesizer.info(f"Starting audio generation for text: \"{text[:50]}...\" with max_new_tokens={max_new_tokens}")

        # --- 1. Prepare Text Input for CFG (Conditional & Unconditional) ---
        cfg_enabled = cfg_scale > 1.001 # Use a small epsilon for strict cfg_scale=1.0 disabling CFG
        input_dict_cfg = self._prepare_text_input_for_inference(text, cfg_enabled=cfg_enabled)
        
        src_tokens_cfg = input_dict_cfg["src_tokens"]
        src_pos_cfg = input_dict_cfg["src_positions"]
        enc_self_attn_mask_cfg = input_dict_cfg["enc_self_attn_mask"]
        dec_cross_attn_mask_step_cfg = input_dict_cfg["dec_cross_attn_mask_step"] 
        current_batch_size = src_tokens_cfg.shape[0] # Will be 2 if CFG, 1 otherwise

        # --- 2. Encoder Pass ---
        text_embeddings = self.model.text_embeddings(src_tokens_cfg)
        encoder_hidden_states = text_embeddings
        # SabdaModel's encoder_layers is an nn.ModuleList, iterate through it.
        for enc_layer_module in self.model.encoder_layers:
            encoder_hidden_states = enc_layer_module.forward( # Ensure this matches your EncoderLayer's forward signature
                x=encoder_hidden_states, 
                src_pos=src_pos_cfg, 
                attn_mask=enc_self_attn_mask_cfg, 
                deterministic=True # No dropout during inference
            )
        encoder_output_cfg = self.model.encoder_norm(encoder_hidden_states)
        logger_synthesizer.debug(f"Encoder output shape: {encoder_output_cfg.shape}")

        # --- 3. Initialize KVCaches for Decoder ---
        S_src = encoder_output_cfg.shape[1] 
        
        cross_attention_caches_list_cfg = []
        for i in range(self.config.model.decoder.n_layer):
            decoder_layer_module = self.model.decoder_layers[i]
            cross_attn_submodule = decoder_layer_module.cross_attention
            
            k_cross_current = cross_attn_submodule.W_K(encoder_output_cfg)
            if src_pos_cfg is not None and hasattr(cross_attn_submodule, 'rope'):
                k_cross_current = cross_attn_submodule.rope(k_cross_current, position=src_pos_cfg)
            k_cross_current = k_cross_current.transpose(1, 2) 
            
            v_cross_current = cross_attn_submodule.W_V(encoder_output_cfg).transpose(1, 2)
            
            # Handle GQA repetition if n_kv_heads < n_query_heads for cross_attn
            # Assuming cross_attn_submodule.n_gqa_groups is available or calculable
            if cross_attn_submodule.n_gqa_groups > 1:
                k_cross_current = k_cross_current.repeat_interleave(cross_attn_submodule.n_gqa_groups, dim=1)
                v_cross_current = v_cross_current.repeat_interleave(cross_attn_submodule.n_gqa_groups, dim=1)

            cross_cache_instance = KVCache(
                n_heads=cross_attn_submodule.n_query_heads,
                max_len=S_src,
                d_heads=cross_attn_submodule.d_heads,
                device=self.device,
                batch_size=current_batch_size,
                k=k_cross_current,
                v=v_cross_current
            )
            cross_cache_instance.current_idx = S_src 
            cross_attention_caches_list_cfg.append(cross_cache_instance)
        logger_synthesizer.debug(f"Initialized {len(cross_attention_caches_list_cfg)} cross-attention KVCaches.")

        self_attention_caches_list_cfg = [
            KVCache(
                n_heads=self.config.model.decoder.n_gqa_heads,
                max_len=max_new_tokens, 
                d_heads=self.config.model.decoder.d_gqa_heads,
                device=self.device,
                batch_size=current_batch_size 
            ) for _ in range(self.config.model.decoder.n_layer)
        ]
        logger_synthesizer.debug(f"Initialized {len(self_attention_caches_list_cfg)} self-attention KVCaches for max_len {max_new_tokens}.")

        # --- 4. Autoregressive Decoding Loop ---
        num_channels = self.config.data.channels
        audio_bos_value = self.config.data.audio_bos_value
        audio_eos_value = self.config.data.audio_eos_value
        delay_pattern_tensor = torch.tensor(self.config.data.delay_pattern, dtype=torch.long, device=self.device)
        
        tgt_tokens_step = torch.full(
            (current_batch_size, 1, num_channels), 
            fill_value=audio_bos_value, 
            dtype=torch.long, 
            device=self.device
        )
        
        collected_conditional_tokens_channels = [] 

        eos_detected_on_primary_channel = False
        max_delay = max(self.config.data.delay_pattern) if self.config.data.delay_pattern else 0
        # Add a few extra steps after EOS on primary channel to allow other channels to complete based on delay
        # Dia uses 30. Let's make it configurable or related to max_delay.
        # For now, let's use max_delay itself as the number of extra steps.
        eos_grace_period_steps = max_delay 
        eos_countdown_steps = -1 

        for step_idx in range(max_new_tokens):
            current_tgt_pos = torch.full(
                (current_batch_size, 1), 
                fill_value=step_idx, 
                dtype=torch.long, 
                device=self.device
            )

            logits_step = self.model.decode_one_step(
                current_tgt_tokens=tgt_tokens_step,
                current_tgt_pos=current_tgt_pos,
                encoder_output=encoder_output_cfg,
                src_pos=src_pos_cfg, 
                dec_cross_attn_mask_step=dec_cross_attn_mask_step_cfg,
                self_attention_caches_list=self_attention_caches_list_cfg,
                cross_attention_caches_list=cross_attention_caches_list_cfg
            ) 
            
            logits_for_sampling = logits_step[:, 0, :, :] 

            if cfg_enabled:
                logits_uncond = logits_for_sampling[0] 
                logits_cond = logits_for_sampling[1]   
                final_logits_to_sample = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            else:
                final_logits_to_sample = logits_for_sampling[0]

            next_dac_tokens_per_channel = self._sample_next_token(
                logits=final_logits_to_sample, 
                temperature=temperature,
                top_p=top_p
            ) 
            
            # This is the token we will actually "commit" and use for the next step's input
            # For EOS logic, we operate on this conditional token.
            committed_next_tokens = next_dac_tokens_per_channel.clone() # Shape (C,)

            if not eos_detected_on_primary_channel and committed_next_tokens[0] == audio_eos_value:
                eos_detected_on_primary_channel = True
                eos_countdown_steps = eos_grace_period_steps 
                logger_synthesizer.info(f"EOS detected on primary channel at step {step_idx}. EOS grace period: {eos_countdown_steps} steps.")

            if eos_countdown_steps > 0: # If we are in the EOS grace period
                step_after_primary_eos = eos_grace_period_steps - eos_countdown_steps
                for c_idx in range(num_channels):
                    if step_after_primary_eos == delay_pattern_tensor[c_idx]:
                        committed_next_tokens[c_idx] = audio_eos_value
                    elif step_after_primary_eos > delay_pattern_tensor[c_idx]:
                        committed_next_tokens[c_idx] = self.config.data.audio_pad_value
                
                eos_countdown_steps -= 1
                if eos_countdown_steps == 0:
                    collected_conditional_tokens_channels.append(committed_next_tokens) # Store the final set of tokens
                    logger_synthesizer.info(f"EOS grace period finished. Stopping generation at step {step_idx}.")
                    break 
            
            collected_conditional_tokens_channels.append(committed_next_tokens)
            
            if eos_detected_on_primary_channel and eos_countdown_steps == 0: # Safety break if already counted down
                 break

            # Prepare input for the *next* decoding step
            tgt_tokens_step = committed_next_tokens.unsqueeze(0).unsqueeze(1).expand(current_batch_size, -1, -1)
        
        logger_synthesizer.info(f"Autoregressive loop finished after {len(collected_conditional_tokens_channels)} actual steps.")

        if not collected_conditional_tokens_channels:
            logger_synthesizer.warning("No tokens were generated.")
            return torch.tensor([], dtype=torch.float32, device=self.device)

        generated_dac_codes_sequence = torch.stack(collected_conditional_tokens_channels, dim=0)
        logger_synthesizer.debug(f"Collected DAC codes sequence shape: {generated_dac_codes_sequence.shape}")

        # --- 5. Post-Process DAC Codes to Waveform ---
        # The _codebook_to_audio helper will handle revert_delay and dac_model.decode()
        waveform = self._codebook_to_audio(generated_dac_codes_sequence)
        
        logger_synthesizer.info("Audio generation finished.")
        return waveform


    # --- Helper Methods ---
    def _prepare_text_input_for_inference(
        self, 
        text: str, 
        cfg_enabled: bool
    ) -> Dict[str, torch.Tensor]:
        logger_synthesizer.debug(f"Preparing text input for: \"{text[:50]}...\", CFG enabled: {cfg_enabled}")

        max_len = self.config.data.txt_len
        text_pad_value = self.config.data.text_pad_value # Should be 0 as per our SabdaConfig
        device = self.device

        # --- 1. Process Conditional Input (the actual text prompt) ---
        # Direct UTF-8 encoding as we're skipping language tags for now.
        byte_text_cond_processed = text.encode('utf-8')
        
        tokens_cond_list = list(byte_text_cond_processed) # Convert byte string to list of integers

        if len(tokens_cond_list) > max_len:
            tokens_cond_list = tokens_cond_list[:max_len]
            logger_synthesizer.warning(f"Input text truncated to {max_len} bytes/tokens.")
        else:
            tokens_cond_list.extend([text_pad_value] * (max_len - len(tokens_cond_list)))
        
        src_tokens_cond = torch.tensor([tokens_cond_list], dtype=torch.long, device=device) # Shape: (1, txt_len)
        src_padding_mask_cond = src_tokens_cond.ne(text_pad_value) # Shape: (1, txt_len)
        enc_self_attn_mask_cond = (src_padding_mask_cond.unsqueeze(1) & src_padding_mask_cond.unsqueeze(2)).unsqueeze(1)
        positions_base = torch.arange(max_len, device=device).unsqueeze(0) # Shape: (1, txt_len)

        if cfg_enabled:
            # --- 2. Create Unconditional Input (if CFG is enabled) ---
            src_tokens_uncond = torch.full_like(src_tokens_cond, fill_value=text_pad_value)
            src_padding_mask_uncond = src_tokens_uncond.ne(text_pad_value)
            enc_self_attn_mask_uncond = (src_padding_mask_uncond.unsqueeze(1) & src_padding_mask_uncond.unsqueeze(2)).unsqueeze(1)
            
            # --- 3. Concatenate for CFG ---
            src_tokens_cfg = torch.cat([src_tokens_uncond, src_tokens_cond], dim=0)
            src_positions_cfg = positions_base.expand(2, -1)
            enc_self_attn_mask_cfg = torch.cat([enc_self_attn_mask_uncond, enc_self_attn_mask_cond], dim=0)
            encoder_output_padding_mask_for_cross_attn = torch.cat([src_padding_mask_uncond, src_padding_mask_cond], dim=0)
        else:
            # --- No CFG ---
            src_tokens_cfg = src_tokens_cond
            src_positions_cfg = positions_base
            enc_self_attn_mask_cfg = enc_self_attn_mask_cond
            encoder_output_padding_mask_for_cross_attn = src_padding_mask_cond

        # --- 4. Create Decoder Cross-Attention Mask (for a single query step) ---
        batch_cfg_size = src_tokens_cfg.shape[0]
        decoder_query_step_exists_mask = torch.ones(batch_cfg_size, 1, dtype=torch.bool, device=device)
        dec_cross_attn_mask_step = (decoder_query_step_exists_mask.unsqueeze(2) & encoder_output_padding_mask_for_cross_attn.unsqueeze(1)).unsqueeze(1)

        logger_synthesizer.debug(f"Prepared text input shapes: tokens {src_tokens_cfg.shape}, "
                                 f"enc_mask {enc_self_attn_mask_cfg.shape}, "
                                 f"cross_mask_step {dec_cross_attn_mask_step.shape}")
        
        return {
            "src_tokens": src_tokens_cfg,
            "src_positions": src_positions_cfg,
            "enc_self_attn_mask": enc_self_attn_mask_cfg,
            "dec_cross_attn_mask_step": dec_cross_attn_mask_step
        }
    

    def _sample_next_token(
        self, 
        logits: torch.Tensor, # Expected shape (num_channels, tgt_vocab_size) after CFG
        temperature: float, 
        top_p: Optional[float]
    ) -> torch.Tensor: # Returns (num_channels,) tensor of sampled token IDs
        """Samples the next token for each channel based on the logits."""
        logger_synthesizer.debug(f"Sampling next token. Logits shape: {logits.shape}, Temp: {temperature}, Top_p: {top_p}")
        
        if logits.ndim != 2: # Expected (C, V)
            raise ValueError(f"Logits for sampling must be 2D (num_channels, vocab_size), got {logits.shape}")

        # Handle greedy decoding case (temperature == 0 or not specified for sampling)
        if temperature == 0.0:
            next_tokens = torch.argmax(logits, dim=-1)
            logger_synthesizer.debug(f"Greedy sampled tokens: {next_tokens}")
            return next_tokens

        # Apply temperature
        logits = logits / temperature

        # Top-p (nucleus) sampling
        if top_p is not None and 0.0 < top_p < 1.0:
            # Calculate probabilities and sort them
            probabilities = torch.softmax(logits, dim=-1) # Shape: (num_channels, vocab_size)
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
            
            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1) # Shape: (num_channels, vocab_size)

            # Identify tokens to remove (those outside the nucleus)
            # Create a mask for tokens to keep (cumulative_probs <= top_p, but keep first if top_p is too small)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask: always keep at least the most probable token in each row
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False # Keep the first token

            # Create a mask for the original logits tensor
            # We need to scatter `sorted_indices_to_remove` back to the original order
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            
            # Mask out non-nucleus tokens by setting their logits to -infinity
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Note: Dia's _sample_next_token also has parameters `use_cfg_filter` and `cfg_filter_top_k`.
        # This `cfg_filter_top_k` is a top-K filtering applied *on the CFG-combined logits* BEFORE top-p or final softmax.
        # We can add this if needed for closer replication, but for now, this implementation covers temp + top-p.

        # Sample from the (potentially modified) logits distribution
        final_probabilities = torch.softmax(logits, dim=-1)
        
        # Multinomial sampling expects 1D or 2D input. If logits are (C,V), multinomial works per row.
        next_tokens = torch.multinomial(final_probabilities, num_samples=1).squeeze(-1) # Shape: (num_channels,)
        
        logger_synthesizer.debug(f"Sampled next tokens (temp={temperature}, top_p={top_p}): {next_tokens}")
        return next_tokens

    def _codebook_to_audio(self, generated_dac_codes_sequence: torch.Tensor) -> torch.Tensor:
        # ... (logging, initial checks, and revert_audio_delay logic remain the same) ...
        logger_synthesizer.debug(f"Converting codebook to audio. Input codes sequence shape: {generated_dac_codes_sequence.shape}")
        
        if generated_dac_codes_sequence.ndim != 2:
             raise ValueError(f"Expected generated_dac_codes_sequence to be 2D (L, C), got {generated_dac_codes_sequence.shape}")
        if generated_dac_codes_sequence.shape[0] == 0:
            logger_synthesizer.warning("Received empty DAC code sequence for audio conversion. Returning empty waveform.")
            return torch.tensor([], dtype=torch.float32, device=self.device)

        L_generated, C_channels = generated_dac_codes_sequence.shape
        
        if C_channels != self.config.data.channels:
            raise ValueError(f"Number of channels in codes ({C_channels}) does not match config ({self.config.data.channels}).")

        # --- 1. Revert Audio Delay ---
        try:
            t_idx_map_revert, gather_map_revert = build_revert_indices(
                B=1, 
                T=L_generated, 
                C=C_channels, 
                delay_pattern=self.config.data.delay_pattern,
                device=self.device 
            )
        except Exception as e:
            logger_synthesizer.error(f"Error in build_revert_indices: {e}", exc_info=True)
            raise

        codes_for_revert = generated_dac_codes_sequence.unsqueeze(0).to(self.device) 
        pad_value_for_revert = 0 

        try:
            reverted_codes_BTC = revert_audio_delay(
                audio_BxTxC=codes_for_revert,
                pad_value=pad_value_for_revert, 
                precomputed_indices=(t_idx_map_revert, gather_map_revert),
                T_original=L_generated 
            )
        except Exception as e:
            logger_synthesizer.error(f"Error in revert_audio_delay: {e}", exc_info=True)
            raise
        
        max_delay = max(self.config.data.delay_pattern) if self.config.data.delay_pattern else 0
        current_reverted_len = reverted_codes_BTC.shape[1]
        if current_reverted_len > max_delay and max_delay > 0 :
             # Trim from the end if there's enough length
            reverted_codes_BTC = reverted_codes_BTC[:, :current_reverted_len - max_delay, :]
            logger_synthesizer.debug(f"Trimmed reverted codes by {max_delay} frames. New shape: {reverted_codes_BTC.shape}")
        
        codes_for_quantizer_input = reverted_codes_BTC # Shape (1, L_reverted_trimmed, C_channels)
        
        if codes_for_quantizer_input.shape[1] == 0: 
            logger_synthesizer.warning("No codes left after reverting delay and trimming for quantizer. Returning empty waveform.")
            return torch.tensor([], dtype=torch.float32, device=self.device)
        
        codes_for_quantizer_input_permuted = codes_for_quantizer_input.permute(0, 2, 1).long() # Shape (B, N_q, T)
        logger_synthesizer.debug(f"Codes prepared for DAC quantizer (permuted) shape: {codes_for_quantizer_input_permuted.shape}")

        # --- NEW: Clamp DAC codes to valid range [0, codebook_size-1] ---
        # Standard DAC codebook size is 1024 (indices 0-1023)
        dac_codebook_min_idx = 0
        dac_codebook_max_idx = 1023 # For a codebook of size 1024

        invalid_mask = (codes_for_quantizer_input_permuted < dac_codebook_min_idx) | \
                       (codes_for_quantizer_input_permuted > dac_codebook_max_idx)
        
        num_invalid_tokens = invalid_mask.sum().item()
        if num_invalid_tokens > 0:
            logger_synthesizer.warning(
                f"Clamping {num_invalid_tokens} token IDs to be within range [{dac_codebook_min_idx}, {dac_codebook_max_idx}]. "
                f"Mapping out-of-range tokens to {dac_codebook_min_idx}."
            )
            # Clamp values. A common strategy is to map to 0 or a known "silence" like token.
            # Dia maps to 0. Let's do the same for now.
            codes_for_quantizer_input_permuted = codes_for_quantizer_input_permuted.masked_fill(invalid_mask, dac_codebook_min_idx)
            # Alternatively, more robust clamping:
            # codes_for_quantizer_input_permuted = torch.clamp(
            #     codes_for_quantizer_input_permuted, min=dac_codebook_min_idx, max=dac_codebook_max_idx
            # )


        # --- 3. Convert Integer Codes to Float Embeddings using DAC's Quantizer ---
        try:
            quantized_embeddings_tuple = self.dac_model.quantizer.from_codes(codes_for_quantizer_input_permuted)
            float_embeddings = quantized_embeddings_tuple[0] 
            logger_synthesizer.debug(f"Float embeddings from quantizer shape: {float_embeddings.shape}")
        except Exception as e:
            logger_synthesizer.error(f"Error during DAC quantizer.from_codes: {e}", exc_info=True)
            logger_synthesizer.error(f"Problematic input to from_codes was shape: {codes_for_quantizer_input_permuted.shape}, dtype: {codes_for_quantizer_input_permuted.dtype}")
            raise

        # --- 4. Decode Float Embeddings to Waveform using DAC's Decoder ---
        try:
            waveform = self.dac_model.decode(float_embeddings) 
            logger_synthesizer.debug(f"Decoded waveform shape from DAC: {waveform.shape}")
            final_waveform = waveform.squeeze(0).squeeze(0) 
            return final_waveform
        except Exception as e:
            logger_synthesizer.error(f"Error during DAC final decoding (from float embeddings): {e}", exc_info=True)
            logger_synthesizer.error(f"Problematic float_embeddings shape: {float_embeddings.shape}, dtype: {float_embeddings.dtype}")
            raise