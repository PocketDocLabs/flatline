#![allow(non_snake_case)]

//! MCP resource management — list, read, subscribe, @mention parsing.
//!
//! Resources are server-provided data (files, URIs, API responses) that
//! can be injected into conversations via `@server:uri` mentions in user
//! messages.
//!
//! # Public API
//! - [`ResourceManager`] — tracks resources across all connected servers
//! - [`parseMentions`] — extract `@server:uri` patterns from text
//!
//! # Dependencies
//! None (uses types from parent module).

use std::collections::{HashMap, HashSet};

/// Information about a single MCP resource.
#[derive(Debug, Clone)]
pub struct ResourceInfo {
    pub serverName: String,
    pub uri: String,
    pub name: String,
    pub description: String,
    pub mimeType: Option<String>,
}

/// Central tracker for MCP resources across all connected servers.
pub struct ResourceManager {
    /// serverName → list of resources.
    resources: HashMap<String, Vec<ResourceInfo>>,
    /// Active subscriptions: (serverName, uri).
    subscriptions: HashSet<(String, String)>,
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            subscriptions: HashSet::new(),
        }
    }

    /// Register resources discovered from a server.
    pub fn registerServer(&mut self, serverName: &str, resources: Vec<ResourceInfo>) {
        tracing::debug!(
            server = %serverName,
            count = resources.len(),
            "registered MCP resources"
        );
        self.resources.insert(serverName.to_string(), resources);
    }

    /// Remove all resources for a server.
    pub fn unregisterServer(&mut self, serverName: &str) {
        self.resources.remove(serverName);
        self.subscriptions.retain(|(s, _)| s != serverName);
    }

    /// Get all resources for a server.
    pub fn serverResources(&self, serverName: &str) -> &[ResourceInfo] {
        self.resources
            .get(serverName)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all resources across all servers.
    pub fn allResources(&self) -> Vec<&ResourceInfo> {
        self.resources.values().flat_map(|v| v.iter()).collect()
    }

    /// Track a subscription to a resource URI.
    pub fn subscribe(&mut self, serverName: &str, uri: &str) {
        self.subscriptions
            .insert((serverName.to_string(), uri.to_string()));
    }

    /// Check if a resource is subscribed.
    pub fn isSubscribed(&self, serverName: &str, uri: &str) -> bool {
        self.subscriptions
            .contains(&(serverName.to_string(), uri.to_string()))
    }

    /// Total number of resources.
    pub fn totalCount(&self) -> usize {
        self.resources.values().map(|v| v.len()).sum()
    }
}

/// A parsed @mention from user text.
#[derive(Debug, Clone, PartialEq)]
pub struct Mention {
    pub serverName: String,
    pub uri: String,
    /// Start byte offset in the original text.
    pub start: usize,
    /// End byte offset in the original text.
    pub end: usize,
}

/// Parse `@server:uri` mentions from text.
///
/// Recognizes the pattern `@servername:resource-uri` where the server name
/// is `[a-zA-Z0-9_-]+` and the URI continues until whitespace or end of string.
pub fn parseMentions(text: &str) -> Vec<Mention> {
    let mut mentions = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'@' {
            let start = i;
            i += 1;

            // Parse server name: [a-zA-Z0-9_-]+
            let serverStart = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'-')
            {
                i += 1;
            }
            let serverEnd = i;

            // Must have a colon separator.
            if i < bytes.len() && bytes[i] == b':' && serverEnd > serverStart {
                i += 1;

                // Parse URI: everything until whitespace.
                let uriStart = i;
                while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                let uriEnd = i;

                if uriEnd > uriStart {
                    mentions.push(Mention {
                        serverName: text[serverStart..serverEnd].to_string(),
                        uri: text[uriStart..uriEnd].to_string(),
                        start,
                        end: uriEnd,
                    });
                }
            }
        } else {
            i += 1;
        }
    }

    mentions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parseMentionsBasic() {
        let mentions = parseMentions("Check @github:repos/flatline/README.md for details");
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].serverName, "github");
        assert_eq!(mentions[0].uri, "repos/flatline/README.md");
    }

    #[test]
    fn parseMentionsMultiple() {
        let mentions = parseMentions("@server1:file.txt and @server2:data.json");
        assert_eq!(mentions.len(), 2);
        assert_eq!(mentions[0].serverName, "server1");
        assert_eq!(mentions[1].serverName, "server2");
    }

    #[test]
    fn parseMentionsNoColon() {
        let mentions = parseMentions("@username is not a mention");
        assert_eq!(mentions.len(), 0);
    }

    #[test]
    fn parseMentionsEmptyUri() {
        let mentions = parseMentions("@server: has no uri");
        // The space after colon means URI is empty.
        assert_eq!(mentions.len(), 0);
    }

    #[test]
    fn parseMentionsHyphenatedServer() {
        let mentions = parseMentions("@my-server:some/resource");
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].serverName, "my-server");
    }

    #[test]
    fn resourceManagerBasics() {
        let mut mgr = ResourceManager::new();
        mgr.registerServer(
            "github",
            vec![ResourceInfo {
                serverName: "github".into(),
                uri: "repos/flatline".into(),
                name: "flatline repo".into(),
                description: "Main repo".into(),
                mimeType: None,
            }],
        );

        assert_eq!(mgr.totalCount(), 1);
        assert_eq!(mgr.serverResources("github").len(), 1);

        mgr.unregisterServer("github");
        assert_eq!(mgr.totalCount(), 0);
    }
}
