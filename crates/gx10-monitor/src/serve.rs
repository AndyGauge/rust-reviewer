//! Runs on gx10: samples periodically in the background and serves the
//! latest sample (`/health`) and recent history (`/samples`) over HTTP, so an
//! external `watch` process can record it without SSHing in.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use tokio::io::AsyncWriteExt;

use crate::sample::{Sample, collect_sample};

pub struct ServeArgs {
    pub bind: SocketAddr,
    pub interval_secs: u64,
    pub process_pattern: String,
    pub log_file: Option<PathBuf>,
    pub history_capacity: usize,
}

#[derive(Clone)]
struct AppState {
    history: Arc<Mutex<VecDeque<Sample>>>,
}

pub async fn run(args: ServeArgs) -> Result<()> {
    let history: Arc<Mutex<VecDeque<Sample>>> =
        Arc::new(Mutex::new(VecDeque::with_capacity(args.history_capacity)));
    let state = AppState {
        history: history.clone(),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/samples", get(samples))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(args.bind)
        .await
        .with_context(|| format!("binding {}", args.bind))?;
    println!("gx10-monitor: serving on http://{}", args.bind);

    let sampler = tokio::spawn(sample_loop(
        args.interval_secs,
        args.process_pattern,
        args.log_file,
        history,
        args.history_capacity,
    ));

    axum::serve(listener, app).await.context("axum server")?;
    sampler.abort();
    Ok(())
}

async fn sample_loop(
    interval_secs: u64,
    process_pattern: String,
    log_file: Option<PathBuf>,
    history: Arc<Mutex<VecDeque<Sample>>>,
    capacity: usize,
) {
    let mut log = match &log_file {
        Some(path) => match tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
        {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!("gx10-monitor: couldn't open log file {path:?}: {e:#}");
                None
            }
        },
        None => None,
    };

    let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
    loop {
        interval.tick().await;
        match collect_sample(&process_pattern).await {
            Ok(sample) => {
                let line = serde_json::to_string(&sample).expect("Sample serializes");
                println!("{line}");
                if let Some(f) = log.as_mut() {
                    if let Err(e) = f.write_all(format!("{line}\n").as_bytes()).await {
                        eprintln!("gx10-monitor: failed writing log file: {e:#}");
                    }
                }

                let mut hist = history.lock().expect("history mutex poisoned");
                if hist.len() >= capacity {
                    hist.pop_front();
                }
                hist.push_back(sample);
            }
            Err(e) => eprintln!("gx10-monitor: sample failed: {e:#}"),
        }
    }
}

async fn health(State(state): State<AppState>) -> Result<Json<Sample>, StatusCode> {
    let hist = state.history.lock().expect("history mutex poisoned");
    hist.back().cloned().map(Json).ok_or(StatusCode::SERVICE_UNAVAILABLE)
}

async fn samples(State(state): State<AppState>) -> Json<Vec<Sample>> {
    let hist = state.history.lock().expect("history mutex poisoned");
    Json(hist.iter().cloned().collect())
}
