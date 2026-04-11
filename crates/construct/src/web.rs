//! Exa-powered web tools — search, fetch, and find-similar.
//!
//! # Public API
//! - [`ExaClient`] — HTTP client for Exa API calls
//! - [`UrlCache`] — 15-minute TTL cache for fetched page content
//! - [`executeSearch`], [`executeFetch`], [`executeSimilar`] — tool handlers
//!
//! # Dependencies
//! `reqwest`, `serde_json`, `tokio`

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::api;
use crate::config::Config;
use crate::message::Message;

const EXA_BASE_URL: &str = "https://api.exa.ai";
const MAX_WEB_CONTENT: usize = 50_000;
const WEB_SIDECAR_THRESHOLD: usize = 12_000;
const WEB_CACHE_TTL_SECS: u64 = 900;
const MAX_SEARCH_RESULTS: usize = 20;
const SEARCH_SNIPPET_CHARS: usize = 1000;
const EXA_TIMEOUT_SECS: u64 = 60;

const NOT_CONFIGURED_MSG: &str = "Web tools not configured. \
    Set web.searchKey in ~/.config/flatline/config.toml \
    or the EXA_API_KEY environment variable. \
    Get a key at https://exa.ai.";

/// Exa API client — separate from the OpenRouter client (no auth leak).
pub struct ExaClient {
    http: reqwest::Client,
}

impl ExaClient {
    /// Build an Exa client. Returns None if the API key is empty.
    pub fn new(apiKey: &str) -> Option<Self> {
        if apiKey.is_empty() {
            return None;
        }

        let mut headers = reqwest::header::HeaderMap::new();
        if let Ok(val) = reqwest::header::HeaderValue::from_str(apiKey) {
            headers.insert("x-api-key", val);
        }

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(EXA_TIMEOUT_SECS))
            .user_agent("Flatline/0.1")
            .build()
            .ok()?;

        Some(Self { http })
    }

    /// POST JSON to an Exa endpoint and return the parsed response.
    async fn post(&self, path: &str, body: &serde_json::Value) -> Result<serde_json::Value, String> {
        let url = format!("{EXA_BASE_URL}{path}");

        let response = self
            .http
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| format!("Exa request failed: {e}"))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| format!("Failed to read Exa response: {e}"))?;

        if !status.is_success() {
            return Err(format!("Exa API error ({status}): {text}"));
        }

        serde_json::from_str(&text).map_err(|e| format!("Failed to parse Exa response: {e}"))
    }
}

/// In-memory URL cache with 15-minute TTL.
pub struct UrlCache {
    entries: HashMap<String, (Instant, String)>,
}

impl UrlCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Get cached content if it exists and hasn't expired.
    pub fn get(&mut self, url: &str) -> Option<&str> {
        // Lazy eviction of expired entries.
        let now = Instant::now();
        self.entries
            .retain(|_, (ts, _)| now.duration_since(*ts).as_secs() < WEB_CACHE_TTL_SECS);

        self.entries.get(url).map(|(_, content)| content.as_str())
    }

    /// Store content in the cache.
    pub fn put(&mut self, url: String, content: String) {
        self.entries.insert(url, (Instant::now(), content));
    }
}

/// Format Exa search/findSimilar results into a readable string.
fn formatExaResults(results: &[serde_json::Value]) -> String {
    let mut output = String::new();

    for (i, result) in results.iter().enumerate() {
        let title = result["title"].as_str().unwrap_or("(no title)");
        let url = result["url"].as_str().unwrap_or("?");
        let text = result["text"].as_str().unwrap_or("");

        // Truncate snippet on a char boundary.
        let snippet = if text.len() > SEARCH_SNIPPET_CHARS {
            let end = text.floor_char_boundary(SEARCH_SNIPPET_CHARS);
            format!("{}...", &text[..end])
        } else {
            text.to_string()
        };

        output.push_str(&format!("{}. {title}\n   {url}\n", i + 1));
        if !snippet.is_empty() {
            // Indent snippet lines.
            for line in snippet.lines() {
                output.push_str(&format!("   {line}\n"));
            }
        }
        output.push('\n');
    }

    if output.is_empty() {
        "No results found.".into()
    } else {
        output
    }
}

/// Normalize a URL: upgrade http to https.
fn normalizeUrl(url: &str) -> String {
    if url.starts_with("http://") {
        format!("https://{}", &url[7..])
    } else {
        url.to_string()
    }
}

/// Check Exa statuses array for errors on a specific URL.
fn checkExaStatus(response: &serde_json::Value) -> Option<String> {
    if let Some(statuses) = response["statuses"].as_array() {
        for status in statuses {
            if status["status"].as_str() == Some("error") {
                let tag = status["error"]["tag"].as_str().unwrap_or("UNKNOWN");
                let code = status["error"]["httpStatusCode"]
                    .as_u64()
                    .map(|c| format!(" (HTTP {c})"))
                    .unwrap_or_default();
                return Some(format!("Failed to fetch: {tag}{code}"));
            }
        }
    }
    None
}

// ── Tool Execution Functions ──

/// Execute webFetch — retrieve a URL's content as markdown.
pub async fn executeFetch(
    exa: &ExaClient,
    cache: &mut UrlCache,
    apiClient: &api::Client,
    config: &Config,
    url: &str,
    prompt: Option<&str>,
    subpages: Option<usize>,
) -> String {
    let url = normalizeUrl(url);

    // Check cache (only for non-subpage fetches).
    if subpages.unwrap_or(0) == 0 {
        if let Some(cached) = cache.get(&url) {
            let content = cached.to_string();
            return maybeSummarize(content, prompt, apiClient, config).await;
        }
    }

    let mut body = serde_json::json!({
        "urls": [url],
        "text": true,
    });

    if let Some(n) = subpages {
        if n > 0 {
            body["subpages"] = serde_json::json!(n);
        }
    }

    let response = match exa.post("/contents", &body).await {
        Ok(r) => r,
        Err(e) => return e,
    };

    // Check for crawl errors.
    if let Some(err) = checkExaStatus(&response) {
        return err;
    }

    let results = response["results"].as_array();
    let mut content = String::new();

    if let Some(results) = results {
        for result in results {
            let pageUrl = result["url"].as_str().unwrap_or("");
            let text = result["text"].as_str().unwrap_or("");

            if !text.is_empty() {
                if results.len() > 1 {
                    content.push_str(&format!("── {pageUrl} ──\n"));
                }
                content.push_str(text);
                content.push('\n');
            }

            // Append subpage content.
            if let Some(subpages) = result["subpages"].as_array() {
                for sub in subpages {
                    let subUrl = sub["url"].as_str().unwrap_or("");
                    let subText = sub["text"].as_str().unwrap_or("");
                    if !subText.is_empty() {
                        content.push_str(&format!("\n── {subUrl} ──\n"));
                        content.push_str(subText);
                        content.push('\n');
                    }
                }
            }
        }
    }

    if content.is_empty() {
        return format!("No content returned for {url}.");
    }

    // Cache main page content (pre-summarization).
    if subpages.unwrap_or(0) == 0 {
        cache.put(url.clone(), content.clone());
    }

    // Truncate before summarization check (on a char boundary).
    if content.len() > MAX_WEB_CONTENT {
        let end = content.floor_char_boundary(MAX_WEB_CONTENT);
        content.truncate(end);
        content.push_str("\n\n... content truncated.");
    }

    maybeSummarize(content, prompt, apiClient, config).await
}

/// Execute webSearch — search the web via Exa.
pub async fn executeSearch(
    exa: &ExaClient,
    query: &str,
    allowedDomains: Option<&[String]>,
    blockedDomains: Option<&[String]>,
    maxResults: Option<usize>,
) -> String {
    let numResults = maxResults.unwrap_or(5).min(MAX_SEARCH_RESULTS);

    let mut body = serde_json::json!({
        "query": query,
        "numResults": numResults,
        "contents": {
            "text": {
                "maxCharacters": SEARCH_SNIPPET_CHARS,
            }
        }
    });

    if let Some(domains) = allowedDomains {
        if !domains.is_empty() {
            body["includeDomains"] = serde_json::json!(domains);
        }
    }

    if let Some(domains) = blockedDomains {
        if !domains.is_empty() {
            body["excludeDomains"] = serde_json::json!(domains);
        }
    }

    let response = match exa.post("/search", &body).await {
        Ok(r) => r,
        Err(e) => return e,
    };

    let results = response["results"].as_array();
    match results {
        Some(r) if !r.is_empty() => formatExaResults(r),
        _ => "No results found.".into(),
    }
}

/// Execute webSimilar — find pages similar to a URL via Exa.
pub async fn executeSimilar(
    exa: &ExaClient,
    url: &str,
    allowedDomains: Option<&[String]>,
    blockedDomains: Option<&[String]>,
    maxResults: Option<usize>,
) -> String {
    let numResults = maxResults.unwrap_or(5).min(MAX_SEARCH_RESULTS);

    let mut body = serde_json::json!({
        "url": url,
        "numResults": numResults,
        "contents": {
            "text": {
                "maxCharacters": SEARCH_SNIPPET_CHARS,
            }
        }
    });

    if let Some(domains) = allowedDomains {
        if !domains.is_empty() {
            body["includeDomains"] = serde_json::json!(domains);
        }
    }

    if let Some(domains) = blockedDomains {
        if !domains.is_empty() {
            body["excludeDomains"] = serde_json::json!(domains);
        }
    }

    let response = match exa.post("/findSimilar", &body).await {
        Ok(r) => r,
        Err(e) => return e,
    };

    let results = response["results"].as_array();
    match results {
        Some(r) if !r.is_empty() => formatExaResults(r),
        _ => format!("No similar pages found for {url}."),
    }
}

/// Return the "not configured" error message.
pub fn notConfiguredError() -> String {
    NOT_CONFIGURED_MSG.into()
}

// ── Sidecar Summarization ──

/// If a prompt is provided and content is large, use the utility model to extract relevant info.
async fn maybeSummarize(
    content: String,
    prompt: Option<&str>,
    apiClient: &api::Client,
    config: &Config,
) -> String {
    let prompt = match prompt {
        Some(p) if !p.is_empty() && content.len() > WEB_SIDECAR_THRESHOLD => p,
        _ => return content,
    };

    let messages = vec![
        Message::System {
            content: "Extract the requested information from this web page. \
                Return only the relevant content, formatted clearly."
                .into(),
        },
        Message::User {
            content: format!(
                "Page content:\n\n{content}\n\n---\n\nExtract: {prompt}"
            ).into(),
        },
    ];

    match apiClient
        .complete(&messages, Some(&config.utility.model))
        .await
    {
        Ok((summary, _usage)) => summary,
        Err(e) => {
            tracing::warn!("Sidecar summarization failed: {e}");
            format!("{content}\n\n(Summarization failed: {e})")
        }
    }
}
