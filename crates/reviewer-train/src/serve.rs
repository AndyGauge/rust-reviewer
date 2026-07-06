//! Candle-backed OpenAI-compatible server — the Rust twin of `train/serve.py`.
//! Loads the 27B (+ LoRA) once and answers `POST /v1/chat/completions` using the
//! Stage-4c cached generation path, so `reviewer-run --endpoint http://box:8000/v1`
//! drives the Rust reviewer over the network exactly as it drives the Python one.
//!
//! Single-stream by design: generation holds a mutex, so concurrent requests
//! serialize onto the one GPU (the same choice `serve.py` was forced into — real
//! request concurrency needs continuous batching, which is a separate build).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use axum::{Json, Router, extract::State, routing::get, routing::post};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::config::Config;
use crate::{chat, generate};

struct ModelState {
    w: HashMap<String, Tensor>,
    cfg: Config,
    tok: Tokenizer,
    device: Device,
    model_name: String,
    default_max_new: usize,
    /// Serializes generation — one request at a time on the single GPU stream.
    gen_lock: Mutex<()>,
}

#[derive(Deserialize)]
struct Msg {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatReq {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<Msg>,
    #[serde(default)]
    max_tokens: Option<usize>,
}

#[derive(Serialize)]
struct RespMsg {
    role: &'static str,
    content: String,
}
#[derive(Serialize)]
struct Choice {
    index: usize,
    message: RespMsg,
    finish_reason: &'static str,
}
#[derive(Serialize)]
struct ChatResp {
    id: String,
    object: &'static str,
    model: String,
    choices: Vec<Choice>,
}

async fn models(State(st): State<Arc<ModelState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{"id": st.model_name, "object": "model"}],
    }))
}

async fn chat_completions(
    State(st): State<Arc<ModelState>>,
    Json(req): Json<ChatReq>,
) -> Json<ChatResp> {
    let system = req
        .messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let user = req
        .messages
        .iter()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let max_new = req.max_tokens.unwrap_or(st.default_max_new);
    let model = req.model.clone().unwrap_or_else(|| st.model_name.clone());

    // Generation is blocking (GPU-bound) — run it off the async runtime, and
    // hold the lock so only one generation touches the GPU at a time.
    let st2 = st.clone();
    let content = tokio::task::spawn_blocking(move || -> Result<String> {
        let _guard = st2.gen_lock.lock().unwrap();
        let prompt = chat::render_prompt(&system, &user);
        let ids = chat::encode(&st2.tok, &prompt)?;
        let eos = chat::eos_ids(&st2.tok);
        let full = generate::greedy_generate_cached(&st2.w, &st2.cfg, &ids, max_new, &eos, &st2.device)?;
        chat::decode(&st2.tok, &full[ids.len()..])
    })
    .await
    .map_err(|e| anyhow::anyhow!("join: {e}"))
    .and_then(|r| r)
    .unwrap_or_else(|e| format!("[serve error] {e:#}"));

    Json(ChatResp {
        id: format!("chatcmpl-{}", now_ms()),
        object: "chat.completion",
        model,
        choices: vec![Choice {
            index: 0,
            message: RespMsg { role: "assistant", content: content.trim().to_string() },
            finish_reason: "stop",
        }],
    })
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
pub fn serve(
    weights: &PathBuf,
    config: Option<&std::path::Path>,
    bf16: bool,
    adapter: Option<&std::path::Path>,
    lora_scale: f64,
    tokenizer: &std::path::Path,
    model_name: String,
    port: u16,
    default_max_new: usize,
) -> Result<()> {
    let (cfg, device, w) = crate::load_model_for_generation(weights, config, bf16, adapter, lora_scale)?;
    let tok = chat::load_tokenizer(tokenizer)?;
    let state = Arc::new(ModelState {
        w,
        cfg,
        tok,
        device,
        model_name,
        default_max_new,
        gen_lock: Mutex::new(()),
    });

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let app = Router::new()
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state);
        let addr = format!("0.0.0.0:{port}");
        println!("reviewer serving on http://{addr}/v1  (Ctrl-C to stop)");
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })
}
