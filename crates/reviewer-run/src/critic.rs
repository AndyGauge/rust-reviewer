//! Stage 3, as a swappable box. The critic is the one model-shaped hole in the
//! harness; everything around it is deterministic. Modeling it as a trait keeps
//! that boundary honest:
//!
//! * [`StubCritic`] — fabricates a placeholder per hunk so the full
//!   capture → render → label flywheel runs with no GPU.
//! * [`HttpCritic`] — hits an OpenAI-compatible `/v1/chat/completions` endpoint
//!   (vLLM, TGI, llama.cpp server, a hosted model — anything that speaks the
//!   protocol) with the *same* `reviewer-core` system prompt the model trained
//!   on. This is what talks to the served LoRA once training frees the box.
//!
//! `review` is async (HTTP), and the trait is used through generics rather than
//! `dyn`, so native async-fn-in-trait works without pulling in `async-trait`.

use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::diff::{Hunk, LineKind};

/// One comment the critic emitted about a hunk, before grounding/persistence.
pub struct RawComment {
    pub body: String,
    /// A line the comment refers to, if one could be parsed. `None` is fine —
    /// grounding only fails on a citation the hunk contradicts.
    pub cited_line: Option<u64>,
}

/// A critic failure, *classified* so the concurrency limiter only backs off on
/// genuine congestion. A 401 (bad key) is not a signal to slow down — it's a
/// signal every request will fail, and feeding it to the limiter would just
/// throttle a doomed run. Only overload throttles.
#[derive(Debug)]
pub enum CriticError {
    /// Overload / backpressure — 429, 503, or a timeout. The congestion signal.
    Overload(String),
    /// Anything else — auth, bad URL, decode. Not a congestion signal.
    Other(anyhow::Error),
}

impl CriticError {
    pub fn is_overload(&self) -> bool {
        matches!(self, CriticError::Overload(_))
    }
}

impl std::fmt::Display for CriticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CriticError::Overload(s) => write!(f, "overload ({s})"),
            CriticError::Other(e) => write!(f, "{e:#}"),
        }
    }
}

/// The review model. Static-dispatch only (no `dyn`), so `review` can be a
/// native `async fn`.
#[allow(async_fn_in_trait)] // used via generics on one task; no Send bound needed
pub trait Critic {
    /// Checkpoint/adapter tag recorded on every finding (e.g. `reviewer-lora@epoch3`).
    fn model_version(&self) -> &str;
    /// Review one hunk. `prompt` is the exact `user_prompt` wire text; `hunk` is
    /// the parsed form (real impls may ignore it; the stub uses it to cite a line).
    async fn review(&self, prompt: &str, hunk: &Hunk) -> Result<Vec<RawComment>, CriticError>;
}

/// Placeholder critic: emits one clearly-marked stub comment per hunk, anchored
/// to the hunk's first added line so grounding + rendering exercise real paths.
/// `model_version` is `"stub"` so nothing downstream mistakes it for real output.
pub struct StubCritic;

impl Critic for StubCritic {
    fn model_version(&self) -> &str {
        "stub"
    }

    async fn review(&self, _prompt: &str, hunk: &Hunk) -> Result<Vec<RawComment>, CriticError> {
        let cited_line = hunk
            .lines
            .iter()
            .find(|l| l.kind == LineKind::Add)
            .and_then(|l| l.new_lineno)
            .or(Some(hunk.new_start));
        Ok(vec![RawComment {
            body: format!(
                "[stub critic — no adapter wired yet] would review hunk `{}`. \
                 Real findings replace this once the LoRA endpoint is connected.",
                hunk.header
            ),
            cited_line,
        }])
    }
}

/// Talks to an OpenAI-compatible chat endpoint. Reconstructs the exact two-turn
/// chat the model trained on — `[system: reviewer_core::SYSTEM, user: prompt]` —
/// so serve-time input matches train-time input.
pub struct HttpCritic {
    http: reqwest::Client,
    /// Base URL including the API version, e.g. `http://spark:8000/v1`.
    endpoint: String,
    /// Served model name (the `model` field in the request).
    model: String,
    api_key: Option<String>,
    /// Tag stamped on every finding — which checkpoint produced it.
    model_version: String,
    temperature: f32,
    max_tokens: u32,
}

impl HttpCritic {
    pub fn new(
        endpoint: &str,
        model: &str,
        api_key: Option<String>,
        model_version: &str,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(180)) // generation can be slow
            .build()?;
        Ok(Self {
            http,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            model_version: model_version.to_string(),
            temperature: 0.2,
            max_tokens: 512,
        })
    }
}

impl Critic for HttpCritic {
    fn model_version(&self) -> &str {
        &self.model_version
    }

    async fn review(&self, prompt: &str, _hunk: &Hunk) -> Result<Vec<RawComment>, CriticError> {
        let body = build_body(&self.model, prompt, self.temperature, self.max_tokens);
        let url = format!("{}/chat/completions", self.endpoint);
        let mut req = self.http.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.map_err(|e| {
            // A timeout reads as backpressure (server too busy to answer in time).
            // A refused/failed connection or DNS/TLS error is a config/availability
            // problem, NOT congestion — don't let it throttle the limiter.
            if e.is_timeout() {
                CriticError::Overload(format!("{e}"))
            } else {
                CriticError::Other(anyhow::Error::new(e).context(format!("POST {url}")))
            }
        })?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let snippet = text.chars().take(300).collect::<String>();
            return Err(if status == 429 || status == 503 {
                CriticError::Overload(format!("{status}: {snippet}"))
            } else {
                CriticError::Other(anyhow::anyhow!("critic endpoint {url} -> {status}: {snippet}"))
            });
        }
        let text = resp.text().await.map_err(|e| CriticError::Other(e.into()))?;
        parse_completion(&text).map_err(CriticError::Other)
    }
}

/// Build the chat-completions request body — the two-turn chat the model trained on.
fn build_body(model: &str, prompt: &str, temperature: f32, max_tokens: u32) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [
            { "role": "system", "content": reviewer_core::SYSTEM },
            { "role": "user", "content": prompt },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": false,
    })
}

/// Pull the assistant text out of a chat-completions response. `cited_line` is
/// left `None`: the model emits prose, not structured citations, so grounding
/// stays a no-op safety net until we add citation extraction. An empty/whitespace
/// reply yields no finding.
///
/// Falls back to `reasoning_content`: some servers (vLLM with a reasoning parser)
/// put the answer there with an empty `content`, which would otherwise read as
/// "no finding."
fn parse_completion(json: &str) -> Result<Vec<RawComment>> {
    #[derive(Deserialize)]
    struct Resp {
        choices: Vec<Choice>,
    }
    #[derive(Deserialize)]
    struct Choice {
        message: Msg,
    }
    #[derive(Deserialize)]
    struct Msg {
        #[serde(default)]
        content: String,
        #[serde(default)]
        reasoning_content: String,
    }
    let resp: Resp = serde_json::from_str(json).context("decoding chat completion")?;
    let Some(msg) = resp.choices.into_iter().next().map(|c| c.message) else {
        return Ok(vec![]);
    };
    let body = if !msg.content.trim().is_empty() {
        msg.content
    } else {
        msg.reasoning_content
    };
    let body = body.trim().to_string();
    if body.is_empty() {
        return Ok(vec![]);
    }
    Ok(vec![RawComment {
        body,
        cited_line: None,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_reconstructs_the_training_chat() {
        let b = build_body("reviewer", "Repository: x\n...", 0.2, 512);
        let msgs = b["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], reviewer_core::SYSTEM);
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(b["stream"], false);
    }

    #[test]
    fn parses_a_normal_completion() {
        let json = r#"{"choices":[{"message":{"role":"assistant",
            "content":"This makes `Foo` public — a back-compat commitment."}}]}"#;
        let out = parse_completion(json).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].body.contains("back-compat"));
        assert_eq!(out[0].cited_line, None);
    }

    #[test]
    fn empty_reply_yields_no_finding() {
        let json = r#"{"choices":[{"message":{"role":"assistant","content":"   "}}]}"#;
        assert!(parse_completion(json).unwrap().is_empty());
        let none = r#"{"choices":[]}"#;
        assert!(parse_completion(none).unwrap().is_empty());
    }
}
