//! Chat-template rendering + tokenization for the 27B reviewer's exact prompt
//! shape: one `system` turn, one `user` turn, `add_generation_prompt=True`,
//! `enable_thinking=False`. No tools, no multi-turn, no vision — so instead of
//! pulling in a jinja engine to render the model's general `chat_template.jinja`,
//! we hardcode the token structure for this one shape (TODO 4a option B) and
//! rely on `VerifyChatTemplate` to catch drift against the real template.
use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

/// Render `system` + `user` into the Qwen3.5/3.6 chat format, thinking disabled:
/// `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
///
/// Matches `tok.apply_chat_template([{system},{user}], enable_thinking=False,
/// add_generation_prompt=True)` byte for byte for this message shape — verified
/// by `VerifyChatTemplate` against a Python-produced oracle.
pub fn render_prompt(system: &str, user: &str) -> String {
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        system.trim(),
        user.trim(),
    )
}

/// Load the 27B's `tokenizer.json`.
pub fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("loading tokenizer {}: {e}", path.display()))
}

/// Tokenize already-rendered chat text. The template's special tokens
/// (`<|im_start|>` etc.) are `added_tokens` in `tokenizer.json`, so the
/// tokenizer's own pre-tokenizer splits on them without extra help — no
/// `add_special_tokens` flag needed here (that flag controls tokenizer-added
/// BOS/EOS wrapping, which this chat format doesn't use).
pub fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

/// Decode generated ids back to text, dropping special tokens (`<|im_end|>`
/// and friends) so the output is just the reviewer's comment.
pub fn decode(tok: &Tokenizer, ids: &[u32]) -> Result<String> {
    tok.decode(ids, true).map_err(|e| anyhow::anyhow!("decoding: {e}"))
}

/// The ids that end generation: `<|im_end|>` (end of turn) and `<|endoftext|>`
/// (the pad/eos fallback) — same pair as `generation_config.json`'s
/// `eos_token_id`, looked up by name so this doesn't drift if vocab ids shift
/// between the 9B and 27B tokenizers.
pub fn eos_ids(tok: &Tokenizer) -> Vec<u32> {
    ["<|im_end|>", "<|endoftext|>"]
        .into_iter()
        .filter_map(|t| tok.token_to_id(t))
        .collect()
}

/// The fixture fed to both sides of the byte-match check: the exact
/// `reviewer-core` system prompt + one concrete diff hunk through
/// `reviewer_core::user_prompt`, serialized so a Python oracle script can
/// render/tokenize the identical strings independently.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ChatFixture {
    pub system: String,
    pub user: String,
}

impl ChatFixture {
    /// One representative hunk — real shape (rustc source, a design-relevant
    /// change), not fetched live, so this check has no network dependency.
    pub fn sample() -> Self {
        let hunk = "\
@@ -412,7 +412,7 @@ impl<'tcx> TyCtxt<'tcx> {
     pub fn is_copy_modulo_regions(self, ty: Ty<'tcx>) -> bool {
-        ty.is_trivially_pure_clone_copy()
+        ty.is_trivially_pure_clone_copy() || self.type_is_copy_modulo_regions(ty)
     }
";
        let system = reviewer_core::SYSTEM.to_string();
        let user = reviewer_core::user_prompt(
            "rust-lang/rust",
            Some(158822),
            "compiler/rustc_middle/src/ty/mod.rs",
            hunk,
        );
        Self { system, user }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))
    }
}
