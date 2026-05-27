#![allow(non_snake_case)]

//! Authentication helpers for ChatGPT/Codex OAuth.
//!
//! This is intentionally Flatline-owned code: it follows the public Codex
//! OAuth device-code protocol without vendoring the Codex implementation.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use base64::Engine;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

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
