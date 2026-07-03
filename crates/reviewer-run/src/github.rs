//! Stage 1: fetch everything a review needs from the GitHub REST API — PR
//! metadata, the unified diff, and existing human comments (both inline review
//! comments and general issue comments). Pure I/O; no model.
//!
//! Auth and headers mirror `reviewer-extract` (bearer token, `X-GitHub-Api-Version`).

use anyhow::{Context, Result, bail};
use reviewer_core::ReviewComment;
use serde::Deserialize;

const API: &str = "https://api.github.com";

pub struct Client {
    http: reqwest::Client,
    token: String,
}

/// PR-level metadata for the report header.
#[derive(Debug, Deserialize)]
pub struct PrMeta {
    pub number: u64,
    pub title: String,
    #[serde(default)]
    pub body: Option<String>,
    #[serde(default)]
    pub user: Option<Author>,
    pub base: Ref,
    #[serde(default)]
    pub changed_files: u32,
    #[serde(default)]
    pub additions: u32,
    #[serde(default)]
    pub deletions: u32,
    #[serde(default)]
    pub html_url: String,
}

#[derive(Debug, Deserialize)]
pub struct Author {
    pub login: String,
}

#[derive(Debug, Deserialize)]
pub struct Ref {
    #[serde(rename = "ref")]
    pub name: String,
}

/// A general (non-inline) PR comment from the issues endpoint.
#[derive(Debug, Deserialize)]
pub struct IssueComment {
    pub body: String,
    #[serde(default)]
    pub user: Option<Author>,
}

/// Everything stage 1 pulls, bundled for the later stages.
pub struct FetchedPr {
    pub meta: PrMeta,
    pub diff: String,
    pub review_comments: Vec<ReviewComment>,
    pub issue_comments: Vec<IssueComment>,
}

impl Client {
    pub fn new(token: String) -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent("rustc-reviewer-run/0.1")
            .build()?;
        Ok(Self { http, token })
    }

    async fn get(&self, url: &str, accept: &str) -> Result<reqwest::Response> {
        let resp = self
            .http
            .get(url)
            .bearer_auth(&self.token)
            .header("Accept", accept)
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("GET {url} -> {status}: {}", body.chars().take(300).collect::<String>());
        }
        Ok(resp)
    }

    pub async fn fetch(&self, repo: &str, pr: u64) -> Result<FetchedPr> {
        let meta = self.pr_meta(repo, pr).await.context("fetching PR metadata")?;
        let diff = self.pr_diff(repo, pr).await.context("fetching PR diff")?;
        let review_comments = self
            .paginated::<ReviewComment>(&format!("{API}/repos/{repo}/pulls/{pr}/comments"))
            .await
            .context("fetching inline review comments")?;
        let issue_comments = self
            .paginated::<IssueComment>(&format!("{API}/repos/{repo}/issues/{pr}/comments"))
            .await
            .context("fetching issue comments")?;
        Ok(FetchedPr {
            meta,
            diff,
            review_comments,
            issue_comments,
        })
    }

    async fn pr_meta(&self, repo: &str, pr: u64) -> Result<PrMeta> {
        let url = format!("{API}/repos/{repo}/pulls/{pr}");
        let body = self
            .get(&url, "application/vnd.github+json")
            .await?
            .text()
            .await?;
        serde_json::from_str(&body).context("decoding PR metadata")
    }

    /// The unified diff, base-relative, via the diff media type.
    async fn pr_diff(&self, repo: &str, pr: u64) -> Result<String> {
        let url = format!("{API}/repos/{repo}/pulls/{pr}");
        Ok(self
            .get(&url, "application/vnd.github.diff")
            .await?
            .text()
            .await?)
    }

    /// Follow `Link: rel="next"` pagination, collecting every page.
    async fn paginated<T: for<'de> Deserialize<'de>>(&self, base: &str) -> Result<Vec<T>> {
        let mut url = Some(format!("{base}?per_page=100"));
        let mut out = Vec::new();
        while let Some(next) = url {
            let resp = self.get(&next, "application/vnd.github+json").await?;
            url = next_link(resp.headers().get("link").and_then(|v| v.to_str().ok()));
            let page: Vec<T> = serde_json::from_str(&resp.text().await?)
                .context("decoding comment page")?;
            if page.is_empty() {
                break;
            }
            out.extend(page);
        }
        Ok(out)
    }
}

/// Extract the `rel="next"` URL from a Link header, if any.
fn next_link(header: Option<&str>) -> Option<String> {
    let header = header?;
    for part in header.split(',') {
        let mut segs = part.split(';');
        let url = segs.next()?.trim().trim_start_matches('<').trim_end_matches('>');
        if segs.any(|s| s.contains("rel=\"next\"")) {
            return Some(url.to_string());
        }
    }
    None
}
