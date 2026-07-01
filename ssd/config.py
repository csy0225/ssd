import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch
from ssd.paths import DEFAULT_TARGET, DEFAULT_DRAFT

@dataclass
class Config:
    model: str = DEFAULT_TARGET
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 1 
    max_model_len: int = 4096 
    gpu_memory_utilization: float = 0.7
    num_gpus: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # spec config args
    draft_hf_config: AutoConfig | None = None
    speculate: bool = False 
    draft: str = DEFAULT_DRAFT
    speculate_k: int = 1
    draft_async: bool = False
    
    # async spec only
    async_fan_out: int = 3
    fan_out_list: list[int] | None = None
    fan_out_list_miss: list[int] | None = None
    sampler_x: float | None = None 
    jit_speculate: bool = False 

    # eagle3
    use_eagle: bool = False 
    eagle_layers: list[int] | None = None   
    d_model_target: int | None = None
    tokenizer_path: str | None = None

    # Debugging
    verbose: bool = False 
    debug_mode: bool = False 
    max_steps: int | None = None

    @property
    def max_blocks(self): 
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size

    def __post_init__(self):
        model = self.model 
        assert os.path.isdir(model)

        assert 1 <= self.num_gpus <= 8 # this codebase only works on one node
        self.hf_config = self._load_hf_config(model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings)
        if self.speculate:
            draft = self.draft
            self.draft_hf_config = self._load_hf_config(draft)
            self.max_model_len = min(
                self.max_model_len, self.draft_hf_config.max_position_embeddings)
            if self.draft_async:
                if self.fan_out_list is None: 
                    self.fan_out_list = [self.async_fan_out] * (self.speculate_k + 1)
                    self.MQ_LEN = sum(self.fan_out_list)
                if self.fan_out_list_miss is None:
                    self.fan_out_list_miss = self.fan_out_list 
                assert sum(self.fan_out_list_miss) == sum(self.fan_out_list), "ERROR in Config: fan_out_list_miss must be the same as fan_out_list"
                
        if self.use_eagle:
            if self.eagle_layers is None:
                L = self.hf_config.num_hidden_layers
                # self.eagle_layers = [3, L//2, L-3]
                self.eagle_layers = [2, L//2, L-3] # [2, 16, 29] outputs, ie. [3, L//2+1, L-2] inputs
                print(f'[Config] just set eagle_layers={self.eagle_layers}', flush=True)
            # Eagle draft must use target's rope_theta (draft config may default to wrong value)
            if self.speculate and self.draft_hf_config is not None:
                target_rope_theta = getattr(self.hf_config, 'rope_theta', 500000.0)
                draft_rope_theta = getattr(self.draft_hf_config, 'rope_theta', 10000.0)
                if target_rope_theta != draft_rope_theta:
                    print(f'[Config] Overriding eagle draft rope_theta: {draft_rope_theta} -> {target_rope_theta}', flush=True)
                    self.draft_hf_config.rope_theta = target_rope_theta
                # Also override max_position_embeddings for correct RoPE cache size
                # NOTE: Do NOT change max_model_len here - it was already correctly capped.
                # Only change draft_hf_config.max_position_embeddings for RoPE.
                target_max_pos = getattr(self.hf_config, 'max_position_embeddings', 8192)
                draft_max_pos = getattr(self.draft_hf_config, 'max_position_embeddings', 2048)
                if target_max_pos != draft_max_pos:
                    print(f'[Config] Overriding eagle draft max_position_embeddings: {draft_max_pos} -> {target_max_pos}', flush=True)
                    self.draft_hf_config.max_position_embeddings = target_max_pos
        
        assert self.max_num_batched_tokens >= self.max_model_len

    def _load_hf_config(self, path: str):
        """Load an HF config, flattening the 'speculators' EAGLE-3 format.

        The speculators checkpoint (e.g. RedHatAI Qwen3-32B eagle3) nests the
        transformer fields under 'transformer_layer_config' and has no top-level
        'model_type', so AutoConfig.from_pretrained fails. Detect that format
        (by the presence of 'transformer_layer_config') and build a flat
        LlamaConfig; otherwise fall back to AutoConfig.
        """
        cfg_path = os.path.join(path, "config.json")
        if os.path.exists(cfg_path):
            import json
            with open(cfg_path) as f:
                ecfg = json.load(f)
            tlc = ecfg.get("transformer_layer_config")
            if tlc is not None:
                from transformers import LlamaConfig
                dc = LlamaConfig(
                    hidden_size=tlc["hidden_size"],
                    intermediate_size=tlc["intermediate_size"],
                    num_hidden_layers=tlc.get("num_hidden_layers", 1),
                    num_attention_heads=tlc["num_attention_heads"],
                    num_key_value_heads=tlc.get("num_key_value_heads", tlc["num_attention_heads"]),
                    head_dim=tlc.get("head_dim"),
                    rms_norm_eps=tlc.get("rms_norm_eps", 1e-6),
                    max_position_embeddings=tlc.get("max_position_embeddings", 4096),
                    rope_theta=tlc.get("rope_theta", 10000.0),
                    vocab_size=tlc["vocab_size"],
                    hidden_act=tlc.get("hidden_act", "silu"),
                    tie_word_embeddings=False,
                )
                dc.draft_vocab_size = ecfg.get("draft_vocab_size", dc.vocab_size)
                # torch_dtype must be a real torch.dtype (engine calls set_default_dtype
                # / .itemsize on it); LlamaConfig leaves it None by default.
                _dt = ecfg.get("torch_dtype") or tlc.get("torch_dtype") or "bfloat16"
                dc.torch_dtype = getattr(torch, _dt) if isinstance(_dt, str) else _dt
                print(f"[Config] loaded speculators eagle3 draft config: hidden={dc.hidden_size}, "
                      f"layers={dc.num_hidden_layers}, heads={dc.num_attention_heads}, "
                      f"kv={dc.num_key_value_heads}, head_dim={dc.head_dim}, "
                      f"draft_vocab={dc.draft_vocab_size}, dtype={dc.torch_dtype}", flush=True)
                return dc
        return AutoConfig.from_pretrained(path)
