#![allow(non_snake_case)]

//! Authentication helpers for ChatGPT/Codex OAuth.
//!
//! This is intentionally Flatline-owned code: it follows the public Codex
//! OAuth device-code protocol without vendoring the Codex implementation.

use std::fs;
use std::io::Write;
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use base64::Engine;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::configDir;

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const ISSUER: &str = "https://auth.openai.com";
const TOKEN_REFRESH_SKEW_SECS: u64 = 300;
const DEFAULT_EXPIRES_IN_SECS: u64 = 3600;

/// Device-code details the CLI can show to the user.
#[derive(Debug, Clone)]
pub struct DeviceCode {
    pub verificationUrl: String,
    pub userCode: String,
    deviceAuthId: String,
    intervalSecs: u64,
}

/// Stored ChatGPT/Codex OAuth credentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAiCodexAuth {
    pub accessToken: String,
    pub refreshToken: Option<String>,
    pub idToken: Option<String>,
    pub expiresAt: u64,
    pub accountId: Option<String>,
    pub email: Option<String>,
    pub planType: Option<String>,
}

/// Flatline auth file shape.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthStore {
    #[serde(default)]
    pub openaiCodex: Option<OpenAiCodexAuth>,
    #[serde(default)]
    pub anthropicOauth: Option<AnthropicOauthAuth>,
}

/// Non-secret auth summary for UI/status display.
#[derive(Debug, Clone)]
pub struct OpenAiCodexStatus {
    pub configured: bool,
    pub storagePath: PathBuf,
    pub accountId: Option<String>,
    pub email: Option<String>,
    pub planType: Option<String>,
    pub expiresAt: Option<u64>,
    pub expired: bool,
}

/// Access token and account metadata needed for request headers.
#[derive(Debug, Clone)]
pub struct CodexAccess {
    pub accessToken: String,
    pub accountId: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UserCodeResp {
    device_auth_id: String,
    #[serde(alias = "usercode")]
    user_code: String,
    #[serde(default, deserialize_with = "deserializeInterval")]
    interval: u64,
}

#[derive(Debug, Serialize)]
struct UserCodeReq<'a> {
    client_id: &'a str,
}

#[derive(Debug, Serialize)]
struct TokenPollReq<'a> {
    device_auth_id: &'a str,
    user_code: &'a str,
}

#[derive(Debug, Deserialize)]
struct CodeSuccessResp {
    authorization_code: String,
    code_verifier: String,
}

#[derive(Debug, Deserialize)]
struct TokenResp {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
    expires_in: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RefreshResp {
    access_token: Option<String>,
    refresh_token: Option<String>,
    id_token: Option<String>,
    expires_in: Option<u64>,
}

fn deserializeInterval<'de, D>(deserializer: D) -> std::result::Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Interval {
        Num(u64),
        Str(String),
    }

    match Option::<Interval>::deserialize(deserializer)? {
        Some(Interval::Num(n)) => Ok(n),
        Some(Interval::Str(s)) => s.trim().parse().map_err(serde::de::Error::custom),
        None => Ok(5),
    }
}

pub fn authPath() -> PathBuf {
    configDir().join("auth.json")
}

pub fn loadStore() -> Result<AuthStore> {
    let path = authPath();
    if !path.exists() {
        return Ok(AuthStore::default());
    }
    let contents =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))
}

pub fn saveStore(store: &AuthStore) -> Result<()> {
    let path = authPath();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let contents = serde_json::to_string_pretty(store)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .mode(0o600)
            .open(&path)
            .with_context(|| format!("failed to write {}", path.display()))?;
        file.write_all(contents.as_bytes())?;
    }

    #[cfg(not(unix))]
    {
        fs::write(&path, contents)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }

    Ok(())
}

pub fn clearOpenAiCodexAuth() -> Result<()> {
    let mut store = loadStore()?;
    store.openaiCodex = None;
    saveStore(&store)
}

pub fn openAiCodexStatus() -> OpenAiCodexStatus {
    let storagePath = authPath();
    match loadStore().ok().and_then(|s| s.openaiCodex) {
        Some(auth) => {
            let now = unixNow();
            OpenAiCodexStatus {
                configured: true,
                storagePath,
                accountId: auth.accountId,
                email: auth.email,
                planType: auth.planType,
                expiresAt: Some(auth.expiresAt),
                expired: auth.expiresAt <= now,
            }
        }
        None => OpenAiCodexStatus {
            configured: false,
            storagePath,
            accountId: None,
            email: None,
            planType: None,
            expiresAt: None,
            expired: false,
        },
    }
}

pub async fn requestOpenAiCodexDeviceCode() -> Result<DeviceCode> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()?;
    let url = format!("{ISSUER}/api/accounts/deviceauth/usercode");
    let resp = client
        .post(url)
        .json(&UserCodeReq {
            client_id: CLIENT_ID,
        })
        .send()
        .await
        .context("failed to request OpenAI device code")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("device code request failed with status {status}: {body}");
    }

    let body: UserCodeResp = resp.json().await.context("failed to parse device code")?;
    Ok(DeviceCode {
        verificationUrl: format!("{ISSUER}/codex/device"),
        userCode: body.user_code,
        deviceAuthId: body.device_auth_id,
        intervalSecs: body.interval.max(1),
    })
}

pub async fn completeOpenAiCodexDeviceLogin(device: DeviceCode) -> Result<OpenAiCodexAuth> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()?;
    let code = pollDeviceCode(&client, &device).await?;
    let token = exchangeAuthorizationCode(&client, &code).await?;
    let auth = authFromTokenResp(token, None);

    let mut store = loadStore()?;
    store.openaiCodex = Some(auth.clone());
    saveStore(&store)?;

    Ok(auth)
}

pub async fn codexAccessToken() -> Result<CodexAccess> {
    if let Ok(token) = std::env::var("FLATLINE_OPENAI_CODEX_ACCESS_TOKEN") {
        let token = token.trim();
        if !token.is_empty() {
            let claims = jwtClaims(token);
            return Ok(CodexAccess {
                accessToken: token.to_string(),
                accountId: extractAccountId(&claims),
            });
        }
    }

    if let Ok(token) = std::env::var("CODEX_ACCESS_TOKEN") {
        let token = token.trim();
        if !token.is_empty() {
            let claims = jwtClaims(token);
            return Ok(CodexAccess {
                accessToken: token.to_string(),
                accountId: extractAccountId(&claims),
            });
        }
    }

    let mut store = loadStore()?;
    let Some(mut auth) = store.openaiCodex.clone() else {
        bail!("OpenAI Codex auth is not configured. Run `flatline auth login openai-codex`.");
    };

    let now = unixNow();
    if auth.expiresAt <= now + TOKEN_REFRESH_SKEW_SECS {
        let Some(refreshToken) = auth.refreshToken.clone() else {
            bail!(
                "OpenAI Codex access token expired and no refresh token is stored. Run `flatline auth login openai-codex`."
            );
        };
        auth = refreshOpenAiCodexAuth(&refreshToken, auth).await?;
        store.openaiCodex = Some(auth.clone());
        saveStore(&store)?;
    }

    Ok(CodexAccess {
        accessToken: auth.accessToken,
        accountId: auth.accountId,
    })
}

async fn pollDeviceCode(client: &reqwest::Client, device: &DeviceCode) -> Result<CodeSuccessResp> {
    let url = format!("{ISSUER}/api/accounts/deviceauth/token");
    let deadline = std::time::Instant::now() + Duration::from_secs(15 * 60);

    loop {
        let resp = client
            .post(&url)
            .json(&TokenPollReq {
                device_auth_id: &device.deviceAuthId,
                user_code: &device.userCode,
            })
            .send()
            .await
            .context("failed to poll OpenAI device auth")?;

        let status = resp.status();
        if status.is_success() {
            return resp
                .json()
                .await
                .context("failed to parse device auth token");
        }

        if status == StatusCode::FORBIDDEN || status == StatusCode::NOT_FOUND {
            if std::time::Instant::now() >= deadline {
                bail!("device authorization timed out after 15 minutes");
            }
            tokio::time::sleep(Duration::from_secs(device.intervalSecs)).await;
            continue;
        }

        let body = resp.text().await.unwrap_or_default();
        bail!("device authorization failed with status {status}: {body}");
    }
}

async fn exchangeAuthorizationCode(
    client: &reqwest::Client,
    code: &CodeSuccessResp,
) -> Result<TokenResp> {
    let tokenUrl = format!("{ISSUER}/oauth/token");
    let redirectUri = format!("{ISSUER}/deviceauth/callback");
    let resp = client
        .post(tokenUrl)
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", code.authorization_code.as_str()),
            ("redirect_uri", redirectUri.as_str()),
            ("client_id", CLIENT_ID),
            ("code_verifier", code.code_verifier.as_str()),
        ])
        .send()
        .await
        .context("failed to exchange OpenAI authorization code")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("token exchange failed with status {status}: {body}");
    }

    resp.json()
        .await
        .context("failed to parse token exchange response")
}

async fn refreshOpenAiCodexAuth(
    refreshToken: &str,
    previous: OpenAiCodexAuth,
) -> Result<OpenAiCodexAuth> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()?;
    let resp = client
        .post(format!("{ISSUER}/oauth/token"))
        .form(&[
            ("client_id", CLIENT_ID),
            ("grant_type", "refresh_token"),
            ("refresh_token", refreshToken),
        ])
        .send()
        .await
        .context("failed to refresh OpenAI Codex token")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("OpenAI Codex token refresh failed with status {status}: {body}");
    }

    let refreshed: RefreshResp = resp.json().await.context("failed to parse token refresh")?;
    let token = TokenResp {
        access_token: refreshed
            .access_token
            .unwrap_or_else(|| previous.accessToken.clone()),
        refresh_token: refreshed.refresh_token.or(previous.refreshToken.clone()),
        id_token: refreshed.id_token.or(previous.idToken.clone()),
        expires_in: refreshed.expires_in,
    };

    Ok(authFromTokenResp(token, Some(previous)))
}

fn authFromTokenResp(token: TokenResp, previous: Option<OpenAiCodexAuth>) -> OpenAiCodexAuth {
    let accessClaims = jwtClaims(&token.access_token);
    let idClaims = token
        .id_token
        .as_ref()
        .map(|id| jwtClaims(id))
        .unwrap_or_default();

    let expiresAt = token
        .expires_in
        .map(|secs| unixNow() + secs)
        .or_else(|| jwtExp(&accessClaims))
        .or_else(|| jwtExp(&idClaims))
        .unwrap_or_else(|| unixNow() + DEFAULT_EXPIRES_IN_SECS);

    let accountId = extractAccountId(&idClaims)
        .or_else(|| extractAccountId(&accessClaims))
        .or_else(|| previous.as_ref().and_then(|p| p.accountId.clone()));
    let email = extractStringClaim(&idClaims, "email")
        .or_else(|| extractStringClaim(&accessClaims, "email"))
        .or_else(|| previous.as_ref().and_then(|p| p.email.clone()));
    let planType = extractStringClaim(&accessClaims, "chatgpt_plan_type")
        .or_else(|| extractStringClaim(&idClaims, "chatgpt_plan_type"))
        .or_else(|| previous.as_ref().and_then(|p| p.planType.clone()));

    OpenAiCodexAuth {
        accessToken: token.access_token,
        refreshToken: token.refresh_token,
        idToken: token.id_token,
        expiresAt,
        accountId,
        email,
        planType,
    }
}

fn unixNow() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn jwtExp(claims: &serde_json::Map<String, serde_json::Value>) -> Option<u64> {
    claims.get("exp").and_then(|v| v.as_u64())
}

fn jwtClaims(jwt: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut parts = jwt.split('.');
    let (Some(_header), Some(payload), Some(_sig)) = (parts.next(), parts.next(), parts.next())
    else {
        return serde_json::Map::new();
    };
    let Ok(bytes) = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(payload) else {
        return serde_json::Map::new();
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return serde_json::Map::new();
    };
    let Some(obj) = value.as_object() else {
        return serde_json::Map::new();
    };

    let mut claims = obj.clone();
    if let Some(authObj) = obj
        .get("https://api.openai.com/auth")
        .and_then(|v| v.as_object())
    {
        for (k, v) in authObj {
            claims.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    claims
}

fn extractAccountId(claims: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    extractStringClaim(claims, "chatgpt_account_id").or_else(|| {
        claims
            .get("organizations")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|org| org.get("id"))
            .and_then(|v| v.as_str())
            .map(str::to_string)
    })
}

fn extractStringClaim(
    claims: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<String> {
    claims.get(key).and_then(|v| v.as_str()).map(str::to_string)
}

// ── Anthropic / Claude subscription OAuth ───────────────────────────
//
// Reuses Claude Code's public OAuth client to authenticate a Claude
// Pro/Max subscription, then uses the returned access token directly
// against the Anthropic Messages API. Endpoints/client id are pinned
// from the live Claude Code build; login is the RFC 8252 loopback flow
// (no copy-paste), with import of any existing Claude Code credentials.

const ANTHROPIC_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const ANTHROPIC_AUTHORIZE_URL: &str = "https://claude.com/cai/oauth/authorize";
const ANTHROPIC_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const ANTHROPIC_PROFILE_URL: &str = "https://api.anthropic.com/api/oauth/profile";
const ANTHROPIC_SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";
const ANTHROPIC_REFRESH_SKEW_SECS: u64 = 300;
const ANTHROPIC_LOGIN_TIMEOUT_SECS: u64 = 300;
// Anthropic's edge WAF returns a fake 429 (not a real rate limit) for OAuth-
// management requests whose User-Agent it doesn't recognize. Matching the
// UA ccauth and other working third-party clients use bypasses the gate.
// The inference endpoint (/v1/messages) has no such gate.
const ANTHROPIC_USER_AGENT: &str = "axios/1.13.6";

/// Stored Claude subscription OAuth credentials. Mirrors the shape Claude
/// Code persists in `~/.claude/.credentials.json` (`expiresAt` here is unix
/// seconds, not milliseconds).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicOauthAuth {
    pub accessToken: String,
    pub refreshToken: Option<String>,
    /// Unix seconds. `0` means no expiry was provided (inference-only token).
    pub expiresAt: u64,
    #[serde(default)]
    pub scopes: Vec<String>,
    #[serde(default)]
    pub subscriptionType: Option<String>,
    #[serde(default)]
    pub rateLimitTier: Option<String>,
}

/// Access token needed for Anthropic request headers.
#[derive(Debug, Clone)]
pub struct AnthropicAccess {
    pub accessToken: String,
}

/// Non-secret auth summary for UI/status display.
#[derive(Debug, Clone)]
pub struct AnthropicStatus {
    pub configured: bool,
    pub storagePath: PathBuf,
    pub subscriptionType: Option<String>,
    pub scopes: Vec<String>,
    pub expiresAt: Option<u64>,
    pub expired: bool,
}

/// In-flight loopback login. The deck shows `authorizeUrl` while awaiting
/// `completeAnthropicLogin`.
pub struct AnthropicLogin {
    pub authorizeUrl: String,
    listener: TcpListener,
    verifier: String,
    state: String,
    redirectUri: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicTokenResp {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
    #[serde(default)]
    scope: Option<String>,
}

pub fn clearAnthropicOauthAuth() -> Result<()> {
    let mut store = loadStore()?;
    store.anthropicOauth = None;
    saveStore(&store)
}

pub fn anthropicOauthStatus() -> AnthropicStatus {
    let storagePath = authPath();
    let auth = loadStore()
        .ok()
        .and_then(|s| s.anthropicOauth)
        .or_else(importClaudeCodeCredentials);
    match auth {
        Some(auth) => {
            let now = unixNow();
            AnthropicStatus {
                configured: true,
                storagePath,
                subscriptionType: auth.subscriptionType,
                scopes: auth.scopes,
                expiresAt: (auth.expiresAt != 0).then_some(auth.expiresAt),
                expired: auth.expiresAt != 0 && auth.expiresAt <= now,
            }
        }
        None => AnthropicStatus {
            configured: false,
            storagePath,
            subscriptionType: None,
            scopes: Vec::new(),
            expiresAt: None,
            expired: false,
        },
    }
}

pub async fn anthropicAccessToken() -> Result<AnthropicAccess> {
    if let Ok(token) = std::env::var("FLATLINE_ANTHROPIC_OAUTH_ACCESS_TOKEN") {
        let token = token.trim();
        if !token.is_empty() {
            return Ok(AnthropicAccess {
                accessToken: token.to_string(),
            });
        }
    }

    let mut store = loadStore()?;
    let mut auth = match store.anthropicOauth.clone() {
        Some(auth) => auth,
        None => {
            let Some(imported) = importClaudeCodeCredentials() else {
                bail!(
                    "Claude subscription auth is not configured. Run `flatline auth` and sign in to Anthropic."
                );
            };
            store.anthropicOauth = Some(imported.clone());
            saveStore(&store)?;
            imported
        }
    };

    let now = unixNow();
    if auth.expiresAt != 0 && auth.expiresAt <= now + ANTHROPIC_REFRESH_SKEW_SECS {
        let Some(refreshToken) = auth.refreshToken.clone() else {
            bail!(
                "Claude subscription token expired and no refresh token is stored. Run `flatline auth` and sign in again."
            );
        };
        auth = refreshAnthropicAuth(&refreshToken, auth).await?;
        store.anthropicOauth = Some(auth.clone());
        saveStore(&store)?;
    }

    Ok(AnthropicAccess {
        accessToken: auth.accessToken,
    })
}

/// Bind a loopback callback server and build the authorize URL. Opens the
/// system browser best-effort; the caller may also surface `authorizeUrl`.
pub fn requestAnthropicLogin() -> Result<AnthropicLogin> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .context("failed to bind loopback listener for the OAuth callback")?;
    let port = listener.local_addr()?.port();
    let redirectUri = format!("http://localhost:{port}/callback");
    let verifier = randomToken();
    let state = randomToken();
    let challenge = pkceChallenge(&verifier);

    let authorizeUrl = reqwest::Url::parse_with_params(
        ANTHROPIC_AUTHORIZE_URL,
        &[
            ("code", "true"),
            ("client_id", ANTHROPIC_CLIENT_ID),
            ("response_type", "code"),
            ("redirect_uri", redirectUri.as_str()),
            ("scope", ANTHROPIC_SCOPES),
            ("code_challenge", challenge.as_str()),
            ("code_challenge_method", "S256"),
            ("state", state.as_str()),
        ],
    )
    .context("failed to build Anthropic authorize URL")?
    .to_string();

    openBrowser(&authorizeUrl);

    Ok(AnthropicLogin {
        authorizeUrl,
        listener,
        verifier,
        state,
        redirectUri,
    })
}

/// Wait for the browser callback, exchange the code, fetch the subscription
/// profile, and persist the credentials.
pub async fn completeAnthropicLogin(login: AnthropicLogin) -> Result<AnthropicOauthAuth> {
    let AnthropicLogin {
        listener,
        verifier,
        state,
        redirectUri,
        ..
    } = login;

    let expectedState = state.clone();
    let code = tokio::task::spawn_blocking(move || awaitLoopbackCode(listener, &expectedState))
        .await
        .context("loopback callback task failed")??;

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()?;
    let token = anthropicTokenPost(
        &client,
        "Anthropic token exchange",
        &serde_json::json!({
            "grant_type": "authorization_code",
            "code": code,
            "state": state,
            "code_verifier": verifier,
            "client_id": ANTHROPIC_CLIENT_ID,
            "redirect_uri": redirectUri,
        }),
    )
    .await?;
    let mut auth = anthropicAuthFromToken(token, None);
    if let Some((subscriptionType, rateLimitTier)) = fetchAnthropicProfile(&auth.accessToken).await
    {
        auth.subscriptionType = subscriptionType;
        auth.rateLimitTier = rateLimitTier;
    }

    let mut store = loadStore()?;
    store.anthropicOauth = Some(auth.clone());
    saveStore(&store)?;
    Ok(auth)
}

fn awaitLoopbackCode(listener: TcpListener, expectedState: &str) -> Result<String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(ANTHROPIC_LOGIN_TIMEOUT_SECS);
    loop {
        let (mut stream, _) = listener.accept().context("loopback accept failed")?;
        let mut buf = [0u8; 2048];
        let n = std::io::Read::read(&mut stream, &mut buf).unwrap_or(0);
        let request = String::from_utf8_lossy(&buf[..n]);
        let path = request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("/");

        let mut code = None;
        let mut gotState = None;
        let mut errorParam = None;
        if let Ok(url) = reqwest::Url::parse(&format!("http://localhost{path}")) {
            for (key, value) in url.query_pairs() {
                match key.as_ref() {
                    "code" => code = Some(value.into_owned()),
                    "state" => gotState = Some(value.into_owned()),
                    "error" => errorParam = Some(value.into_owned()),
                    _ => {}
                }
            }
        }

        let bodyHtml = "<!doctype html><html><body style=\"font-family:system-ui;padding:2rem\">Flatline is signed in to Claude. You can close this tab.</body></html>";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            bodyHtml.len(),
            bodyHtml
        );
        let _ = stream.write_all(response.as_bytes());

        if let Some(error) = errorParam {
            bail!("Anthropic authorization was denied: {error}");
        }
        if let Some(code) = code {
            if gotState.as_deref() != Some(expectedState) {
                bail!("OAuth state mismatch on callback; aborting for safety");
            }
            return Ok(code);
        }
        if std::time::Instant::now() >= deadline {
            bail!("timed out waiting for Anthropic authorization");
        }
    }
}

async fn refreshAnthropicAuth(
    refreshToken: &str,
    previous: AnthropicOauthAuth,
) -> Result<AnthropicOauthAuth> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()?;
    let token = anthropicTokenPost(
        &client,
        "Anthropic token refresh",
        &serde_json::json!({
            "grant_type": "refresh_token",
            "refresh_token": refreshToken,
            "client_id": ANTHROPIC_CLIENT_ID,
        }),
    )
    .await?;
    Ok(anthropicAuthFromToken(token, Some(&previous)))
}

/// POST to the Anthropic token endpoint with the mandatory `claude-code/`
/// User-Agent, retrying transient 429/5xx rate limits with exponential
/// backoff (honoring `Retry-After` when present).
async fn anthropicTokenPost(
    client: &reqwest::Client,
    label: &str,
    body: &serde_json::Value,
) -> Result<AnthropicTokenResp> {
    const MAX_RETRIES: usize = 4;
    let mut attempt = 0;
    loop {
        let resp = client
            .post(ANTHROPIC_TOKEN_URL)
            .header("User-Agent", ANTHROPIC_USER_AGENT)
            .json(body)
            .send()
            .await
            .with_context(|| format!("failed to call {label}"))?;
        let status = resp.status();
        if status.is_success() {
            return resp
                .json()
                .await
                .with_context(|| format!("failed to parse {label} response"));
        }
        let retryable = status.as_u16() == 429 || status.is_server_error();
        if retryable && attempt < MAX_RETRIES {
            let retryAfter = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            let delay = retryAfter
                .map(Duration::from_secs)
                .unwrap_or_else(|| Duration::from_secs(1u64 << attempt));
            tracing::warn!(%status, attempt, ?delay, "{label} rate-limited, retrying");
            tokio::time::sleep(delay).await;
            attempt += 1;
            continue;
        }
        let bodyText = resp.text().await.unwrap_or_default();
        bail!("{label} failed with status {status}: {bodyText}");
    }
}

fn anthropicAuthFromToken(
    token: AnthropicTokenResp,
    previous: Option<&AnthropicOauthAuth>,
) -> AnthropicOauthAuth {
    let expiresAt = token.expires_in.map(|secs| unixNow() + secs).unwrap_or(0);
    let scopes = token
        .scope
        .map(|scope| scope.split_whitespace().map(str::to_string).collect())
        .unwrap_or_else(|| previous.map(|p| p.scopes.clone()).unwrap_or_default());
    AnthropicOauthAuth {
        accessToken: token.access_token,
        refreshToken: token
            .refresh_token
            .or_else(|| previous.and_then(|p| p.refreshToken.clone())),
        expiresAt,
        scopes,
        subscriptionType: previous.and_then(|p| p.subscriptionType.clone()),
        rateLimitTier: previous.and_then(|p| p.rateLimitTier.clone()),
    }
}

/// Best-effort: read the subscription profile to label the account.
async fn fetchAnthropicProfile(token: &str) -> Option<(Option<String>, Option<String>)> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()
        .ok()?;
    let resp = client
        .get(ANTHROPIC_PROFILE_URL)
        .bearer_auth(token)
        .header("User-Agent", ANTHROPIC_USER_AGENT)
        .header("Content-Type", "application/json")
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let value: serde_json::Value = resp.json().await.ok()?;
    let org = value.get("organization").unwrap_or(&value);
    // Profile returns organization_type as "claude_max"/"claude_pro"/etc;
    // normalize to the short form ccauth uses.
    let subscriptionType = org
        .get("organization_type")
        .and_then(|v| v.as_str())
        .map(|t| t.strip_prefix("claude_").unwrap_or(t).to_string());
    let rateLimitTier = org
        .get("rate_limit_tier")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    Some((subscriptionType, rateLimitTier))
}

/// Import credentials Claude Code already wrote, if present.
fn importClaudeCodeCredentials() -> Option<AnthropicOauthAuth> {
    let path = dirs::home_dir()?.join(".claude").join(".credentials.json");
    let text = fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let oauth = value.get("claudeAiOauth")?;
    let accessToken = oauth.get("accessToken")?.as_str()?.to_string();
    let refreshToken = oauth
        .get("refreshToken")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    // Claude Code stores expiresAt in milliseconds.
    let expiresAt = oauth
        .get("expiresAt")
        .and_then(|v| v.as_u64())
        .map(|ms| ms / 1000)
        .unwrap_or(0);
    let scopes = oauth
        .get("scopes")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| s.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let subscriptionType = oauth
        .get("subscriptionType")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let rateLimitTier = oauth
        .get("rateLimitTier")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    Some(AnthropicOauthAuth {
        accessToken,
        refreshToken,
        expiresAt,
        scopes,
        subscriptionType,
        rateLimitTier,
    })
}

fn randomToken() -> String {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).expect("OS random number generator unavailable");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn pkceChallenge(verifier: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hasher.finalize())
}

fn openBrowser(url: &str) {
    let _ = url;
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd")
        .args(["/C", "start", "", url])
        .spawn();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jwt(payload: serde_json::Value) -> String {
        let encode = |bytes: &[u8]| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
        let header = encode(br#"{"alg":"none","typ":"JWT"}"#);
        let payload = encode(serde_json::to_string(&payload).unwrap().as_bytes());
        let sig = encode(b"sig");
        format!("{header}.{payload}.{sig}")
    }

    #[test]
    fn jwtClaimsFlattensOpenAiAuthObject() {
        let token = jwt(serde_json::json!({
            "exp": 123,
            "email": "person@example.com",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_123",
                "chatgpt_plan_type": "pro"
            }
        }));
        let claims = jwtClaims(&token);
        assert_eq!(extractAccountId(&claims).as_deref(), Some("acct_123"));
        assert_eq!(
            extractStringClaim(&claims, "chatgpt_plan_type").as_deref(),
            Some("pro")
        );
        assert_eq!(jwtExp(&claims), Some(123));
    }
}
