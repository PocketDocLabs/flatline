//! JSON schema sanitization and tool name qualification.
//!
//! Cleans up MCP tool schemas for safe LLM consumption (real-world servers
//! produce broken schemas) and handles the `mcp__server__tool` naming convention.
//!
//! # Public API
//! - [`sanitizeJsonSchema`] — recursive schema cleanup
//! - [`qualifyName`] — build qualified tool name
//! - [`sanitizeName`] — clean name component
//! - [`splitQualifiedName`] — reverse lookup
//!
//! # Dependencies
//! `serde_json`, `sha1_smol`

use serde_json::Value;

const MCP_PREFIX: &str = "mcp";
const DELIMITER: &str = "__";
const MAX_QUALIFIED_LEN: usize = 64;

/// Build a qualified tool name: `mcp__{server}__{tool}`.
///
/// Both components are sanitized to `[a-zA-Z0-9_-]`. If the result exceeds
/// 64 chars, the tool portion is truncated and a SHA-1 suffix appended.
pub fn qualifyName(serverName: &str, toolName: &str) -> String {
    let server = sanitizeName(serverName);
    let tool = sanitizeName(toolName);
    let raw = format!("{MCP_PREFIX}{DELIMITER}{server}{DELIMITER}{tool}");

    if raw.len() <= MAX_QUALIFIED_LEN {
        return raw;
    }

    // Truncate and append hash to stay within limit.
    let hash = sha1Hex(&raw);
    let hashSuffix = &hash[..11];
    // prefix = "mcp__{server}__" + as much of tool as fits + "_" + 11-char hash
    let prefixLen = MCP_PREFIX.len() + DELIMITER.len() + server.len() + DELIMITER.len();
    let available = MAX_QUALIFIED_LEN - prefixLen - 1 - 11; // 1 for underscore separator
    let truncatedTool = &tool[..available.min(tool.len())];
    format!("{MCP_PREFIX}{DELIMITER}{server}{DELIMITER}{truncatedTool}_{hashSuffix}")
}

/// Split a qualified name back into `(serverName, toolName)`.
///
/// Returns `None` if the name doesn't match the `mcp__server__tool` pattern.
pub fn splitQualifiedName(name: &str) -> Option<(String, String)> {
    let rest = name.strip_prefix(MCP_PREFIX)?.strip_prefix(DELIMITER)?;
    let delimIdx = rest.find(DELIMITER)?;
    let server = &rest[..delimIdx];
    let tool = &rest[delimIdx + DELIMITER.len()..];
    if server.is_empty() || tool.is_empty() {
        return None;
    }
    Some((server.to_string(), tool.to_string()))
}

/// Check if a tool name is an MCP-qualified name.
pub fn isMcpTool(name: &str) -> bool {
    name.starts_with("mcp__") || name == "mcpToolSearch"
}

/// Sanitize a name component: keep only `[a-zA-Z0-9_-]`, replace everything else.
pub fn sanitizeName(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Validate a server name: must match `[a-zA-Z0-9_-]+`.
pub fn validateServerName(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Server name cannot be empty".into());
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(format!(
            "Server name \"{name}\" contains invalid characters. \
             Only [a-zA-Z0-9_-] are allowed."
        ));
    }
    Ok(())
}

fn sha1Hex(input: &str) -> String {
    let hash = sha1_smol::Sha1::from(input).digest();
    hash.bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
}

/// Recursively sanitize a JSON schema for safe LLM consumption.
///
/// Handles real-world MCP server schema brokenness:
/// - Boolean schemas → proper schema objects
/// - Missing `type` inferred from context
/// - Array `type` fields → pick first recognized
/// - Missing `properties` on objects → empty `{}`
/// - Missing `items` on arrays → `{ "type": "string" }`
/// - Recurses into nested schema structures
pub fn sanitizeJsonSchema(schema: &mut Value) {
    // Boolean schemas: true → accept anything, false → accept nothing.
    if schema.is_boolean() {
        *schema = serde_json::json!({ "type": "string" });
        return;
    }

    let obj = match schema.as_object_mut() {
        Some(o) => o,
        None => return,
    };

    // Normalize array type fields: ["string", "null"] → "string".
    if let Some(typeVal) = obj.get("type").cloned() {
        if let Some(arr) = typeVal.as_array() {
            let first = arr
                .iter()
                .find(|v| v.as_str().is_some_and(|s| s != "null"))
                .cloned()
                .unwrap_or(Value::String("string".into()));
            obj.insert("type".into(), first);
        }
    }

    // Infer missing type from context.
    if !obj.contains_key("type") {
        let inferred = if obj.contains_key("properties")
            || obj.contains_key("required")
            || obj.contains_key("additionalProperties")
        {
            "object"
        } else if obj.contains_key("items") || obj.contains_key("prefixItems") {
            "array"
        } else if obj.contains_key("enum")
            || obj.contains_key("const")
            || obj.contains_key("format")
        {
            "string"
        } else if obj.contains_key("minimum")
            || obj.contains_key("maximum")
            || obj.contains_key("multipleOf")
            || obj.contains_key("exclusiveMinimum")
            || obj.contains_key("exclusiveMaximum")
        {
            "number"
        } else {
            // Default: don't inject a type — leave it flexible.
            ""
        };
        if !inferred.is_empty() {
            obj.insert("type".into(), Value::String(inferred.into()));
        }
    }

    // Ensure objects have properties.
    if obj.get("type").and_then(|v| v.as_str()) == Some("object") {
        if !obj.contains_key("properties") {
            obj.insert("properties".into(), serde_json::json!({}));
        }
    }

    // Ensure arrays have items.
    if obj.get("type").and_then(|v| v.as_str()) == Some("array") {
        if !obj.contains_key("items") && !obj.contains_key("prefixItems") {
            obj.insert("items".into(), serde_json::json!({ "type": "string" }));
        }
    }

    // Recurse into properties.
    if let Some(props) = obj.get_mut("properties") {
        if let Some(propsObj) = props.as_object_mut() {
            for (_, propSchema) in propsObj.iter_mut() {
                sanitizeJsonSchema(propSchema);
            }
        }
    }

    // Recurse into items.
    if let Some(items) = obj.get_mut("items") {
        sanitizeJsonSchema(items);
    }

    // Recurse into prefixItems.
    if let Some(prefixItems) = obj.get_mut("prefixItems") {
        if let Some(arr) = prefixItems.as_array_mut() {
            for item in arr.iter_mut() {
                sanitizeJsonSchema(item);
            }
        }
    }

    // Recurse into additionalProperties (when it's a schema, not a bool).
    if let Some(additional) = obj.get_mut("additionalProperties") {
        if additional.is_object() {
            sanitizeJsonSchema(additional);
        }
    }

    // Recurse into composition keywords.
    for keyword in &["oneOf", "anyOf", "allOf"] {
        if let Some(arr) = obj.get_mut(*keyword) {
            if let Some(variants) = arr.as_array_mut() {
                for variant in variants.iter_mut() {
                    sanitizeJsonSchema(variant);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn qualifyNameBasic() {
        assert_eq!(
            qualifyName("github", "search_repos"),
            "mcp__github__search_repos"
        );
    }

    #[test]
    fn qualifyNameSanitizes() {
        assert_eq!(
            qualifyName("my.server", "do/thing"),
            "mcp__my_server__do_thing"
        );
    }

    #[test]
    fn qualifyNameTruncatesLong() {
        let long = "a".repeat(100);
        let result = qualifyName("server", &long);
        assert!(result.len() <= MAX_QUALIFIED_LEN);
        assert!(result.starts_with("mcp__server__"));
    }

    #[test]
    fn splitQualifiedNameRoundtrip() {
        let qualified = qualifyName("github", "search_repos");
        let (server, tool) = splitQualifiedName(&qualified).unwrap();
        assert_eq!(server, "github");
        assert_eq!(tool, "search_repos");
    }

    #[test]
    fn splitQualifiedNameRejectsInvalid() {
        assert!(splitQualifiedName("shell").is_none());
        assert!(splitQualifiedName("mcp__").is_none());
        assert!(splitQualifiedName("mcp____tool").is_none());
    }

    #[test]
    fn isMcpToolDetects() {
        assert!(isMcpTool("mcp__github__search"));
        assert!(isMcpTool("mcpToolSearch"));
        assert!(!isMcpTool("shell"));
        assert!(!isMcpTool("readFile"));
    }

    #[test]
    fn validateServerNameAccepts() {
        assert!(validateServerName("github").is_ok());
        assert!(validateServerName("my-server").is_ok());
        assert!(validateServerName("server_1").is_ok());
    }

    #[test]
    fn validateServerNameRejects() {
        assert!(validateServerName("").is_err());
        assert!(validateServerName("my.server").is_err());
        assert!(validateServerName("has space").is_err());
    }

    #[test]
    fn sanitizeBooleanSchema() {
        let mut schema = json!(true);
        sanitizeJsonSchema(&mut schema);
        assert_eq!(schema, json!({ "type": "string" }));
    }

    #[test]
    fn sanitizeMissingTypeInferred() {
        let mut schema = json!({ "properties": { "name": { "type": "string" } } });
        sanitizeJsonSchema(&mut schema);
        assert_eq!(schema["type"], "object");
    }

    #[test]
    fn sanitizeArrayTypeField() {
        let mut schema = json!({ "type": ["string", "null"] });
        sanitizeJsonSchema(&mut schema);
        assert_eq!(schema["type"], "string");
    }

    #[test]
    fn sanitizeMissingProperties() {
        let mut schema = json!({ "type": "object" });
        sanitizeJsonSchema(&mut schema);
        assert_eq!(schema["properties"], json!({}));
    }

    #[test]
    fn sanitizeMissingItems() {
        let mut schema = json!({ "type": "array" });
        sanitizeJsonSchema(&mut schema);
        assert_eq!(schema["items"], json!({ "type": "string" }));
    }

    #[test]
    fn sanitizeRecursesIntoProperties() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "nested": { "properties": { "x": { "type": "string" } } }
            }
        });
        sanitizeJsonSchema(&mut schema);
        assert_eq!(schema["properties"]["nested"]["type"], "object");
    }

    #[test]
    fn sanitizeRecursesIntoComposition() {
        let mut schema = json!({
            "oneOf": [
                { "properties": { "a": { "type": "string" } } },
                true
            ]
        });
        sanitizeJsonSchema(&mut schema);
        assert_eq!(schema["oneOf"][0]["type"], "object");
        assert_eq!(schema["oneOf"][1], json!({ "type": "string" }));
    }
}
