//! Model dimensions, read from a HF `config.json`, so the same candle code runs
//! the 9B (verification) and the 27B (the actual reviewer).

use std::path::Path;

use candle_core::Result;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub full_attn_interval: usize,
    pub eps: f64,
    // Gated DeltaNet (linear-attention) layers
    pub nv: usize, // linear_num_value_heads
    pub nk: usize, // linear_num_key_heads
    pub dk: usize, // linear_key_head_dim
    pub dv: usize, // linear_value_head_dim
    pub conv_kernel: usize,
    // Full-attention layers
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    // RoPE (text rotary — mrope collapses to plain RoPE for text-only input,
    // see reviewer-train::rope).
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
}

impl Config {
    pub fn key_dim(&self) -> usize {
        self.nk * self.dk
    }
    pub fn value_dim(&self) -> usize {
        self.nv * self.dv
    }
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
    /// Layer `i` is full-attention when `(i+1)` is a multiple of the interval.
    pub fn is_full_attn(&self, i: usize) -> bool {
        (i + 1) % self.full_attn_interval == 0
    }
    /// Rotary dim: only the first `head_dim * partial_rotary_factor` dims of
    /// each head are rotated (the rest pass through RoPE unchanged).
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }

    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(candle_core::Error::wrap)?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(candle_core::Error::wrap)?;
        // dims may live at the top level or under "text_config"
        let t = v.get("text_config").unwrap_or(&v);
        let get = |k: &str| -> Result<u64> {
            t.get(k)
                .or_else(|| v.get(k))
                .and_then(|x| x.as_u64())
                .ok_or_else(|| candle_core::Error::Msg(format!("config missing {k}")))
        };
        let getf = |k: &str, default: f64| -> f64 {
            t.get(k).or_else(|| v.get(k)).and_then(|x| x.as_f64()).unwrap_or(default)
        };
        // rope_theta / partial_rotary_factor live under text_config.rope_parameters.
        let rope = t.get("rope_parameters").or_else(|| v.get("rope_parameters"));
        let getf_rope = |k: &str, default: f64| -> f64 {
            rope.and_then(|r| r.get(k)).and_then(|x| x.as_f64()).unwrap_or(default)
        };
        Ok(Config {
            hidden: get("hidden_size")? as usize,
            n_layers: get("num_hidden_layers")? as usize,
            vocab: get("vocab_size")? as usize,
            full_attn_interval: get("full_attention_interval")? as usize,
            eps: getf("rms_norm_eps", 1e-6),
            nv: get("linear_num_value_heads")? as usize,
            nk: get("linear_num_key_heads")? as usize,
            dk: get("linear_key_head_dim")? as usize,
            dv: get("linear_value_head_dim")? as usize,
            conv_kernel: get("linear_conv_kernel_dim")? as usize,
            n_heads: get("num_attention_heads")? as usize,
            n_kv_heads: get("num_key_value_heads")? as usize,
            head_dim: get("head_dim")? as usize,
            rope_theta: getf_rope("rope_theta", 10_000_000.0),
            partial_rotary_factor: getf_rope("partial_rotary_factor", 0.25),
        })
    }

    /// Hardcoded 9B config for the component oracles (which predate config.json loading).
    pub fn qwen9b() -> Self {
        Config {
            hidden: 4096, n_layers: 32, vocab: 248320, full_attn_interval: 4, eps: 1e-6,
            nv: 32, nk: 16, dk: 128, dv: 128, conv_kernel: 4,
            n_heads: 16, n_kv_heads: 4, head_dim: 256,
            rope_theta: 10_000_000.0, partial_rotary_factor: 0.25,
        }
    }
}
