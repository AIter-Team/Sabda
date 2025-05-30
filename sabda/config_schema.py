import os
import logging
from pydantic import BaseModel, Field, BeforeValidator
from typing import List, Annotated

class DataConfig(BaseModel): #
    """Configuration for the data processing pipeline
    
    Attributes:
        txt_len: Maximum length of the text (must be multiple of 128)
        audio_len: Maximum length of the audio (must be multiple of 128)
        channels: Number of channels in the audio
        text_pad_value: Padding value for the text
        audio_pad_value: Padding value for the audio
        audio_bos_value: Value to represent the beginning of the audio
        audio_eos_value: Value to represent the end of the audio
        delay_pattern: List of delay values for each audio channel.
        dataset_default_path: Default base path to the dataset directory.
    """

    txt_len: Annotated[int, BeforeValidator(lambda x: (x + 127) // 128 * 128)] = Field(gt=0, multiple_of=128)
    audio_len: Annotated[int, BeforeValidator(lambda x: (x + 127) //  128 * 128)] = Field(gt=0, multiple_of=128)
    channels: int = Field(default=9, gt=0, multiple_of=1)
    text_pad_value: int = Field(default=0)
    audio_eos_value: int = Field(default=1024)
    audio_pad_value: int = Field(default=1025)
    audio_bos_value: int = Field(default=1026)
    delay_pattern: List[Annotated[int, Field(ge=0)]] = Field(default_factory=lambda: [0, 8, 9, 10, 11, 12, 13, 14, 15])

    dataset_path: str = Field(description="Base path to the dataset directory. This field is required.")

    def __hash__(self) -> int: #
        return hash(
            (
                self.txt_len,
                self.audio_len,
                self.channels,
                self.text_pad_value,
                self.audio_eos_value,
                self.audio_pad_value,
                self.audio_bos_value,
                tuple(self.delay_pattern),
                self.dataset_default_path, # Tambahkan ke hash
            )
        )

class EncoderConfig(BaseModel, frozen=True): #
    n_layer: int = Field(gt=0)
    d_embd: int = Field(gt=0)
    d_ff: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    d_heads: int = Field(gt=0)
    ff_activations: List[str] = Field(default=['silu', 'linear'])
    use_pre_norm: bool = Field(default=False)

class DecoderConfig(BaseModel, frozen=True): #
    n_layer: int = Field(gt=0)
    d_embd: int = Field(gt=0)
    d_ff: int = Field(gt=0)
    n_gqa_heads: int = Field(gt=0)
    d_gqa_heads: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    n_cross_heads: int = Field(gt=0)
    d_cross_heads: int = Field(gt=0)
    ff_activations: List[str] = Field(default=['silu', 'linear'])
    use_pre_norm: bool = Field(default=False)

class CoreModelConfig(BaseModel, frozen=True): #
    encoder: EncoderConfig
    decoder: DecoderConfig
    src_vocab_size: int = Field(default=256, gt=0)
    tgt_vocab_size: int = Field(default=1028, gt=0)
    norm_eps: float = Field(default=1e-5, ge=0.0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    weight_dtype: str = Field(default="float32", description="Weight Precision")
    min_rope_timescale: int = Field(default=1, description="Timescale For global Attention")
    max_rope_timescale: int = Field(default=10_000, description="Timescale For global Attention")

class TrainingConfig(BaseModel, frozen=True): #
    dtype: str = Field(default="bfloat16", description="Activation precision")
    logits_dot_in_fp32: bool = Field(default=False)

class SabdaConfig(BaseModel, frozen=True): #
    version: str = Field(default="1.0.1") # Naikkan versi config jika ada perubahan signifikan
    data: DataConfig
    model: CoreModelConfig
    train_args: TrainingConfig

    def save(self, path: str) -> None: #
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config_json = self.model_dump_json(indent=2)
        with open(path, "w", encoding="utf-8") as f:
            f.write(config_json)

    @classmethod
    def load(cls, path: str) -> "SabdaConfig | None": #
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return cls.model_validate_json(content)
        except FileNotFoundError:
            logger_config = logging.getLogger(__name__) # Dapatkan logger di sini jika ingin log error
            logger_config.error(f"File konfigurasi SabdaConfig tidak ditemukan di: {path}")
            return None
        except Exception as e: # pylint: disable=broad-except
            logger_config = logging.getLogger(__name__)
            logger_config.error(f"Error saat memuat atau memvalidasi SabdaConfig dari {path}: {e}", exc_info=True)
            raise # Lemparkan kembali error setelah logging agar masalahnya jelas