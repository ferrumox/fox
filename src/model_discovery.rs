// Multi-path GGUF model discovery across well-known directories.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF"

#[derive(Debug, Clone)]
pub struct DiscoveredModel {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub source: ModelSource,
    pub shard_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    Ferrumox,
    HuggingFace,
    Ollama,
    LmStudio,
    Custom(PathBuf),
}

impl fmt::Display for ModelSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ferrumox => write!(f, "ferrumox"),
            Self::HuggingFace => write!(f, "huggingface"),
            Self::Ollama => write!(f, "ollama"),
            Self::LmStudio => write!(f, "lmstudio"),
            Self::Custom(_) => write!(f, "custom"),
        }
    }
}

pub fn discover_models(extra_dirs: &[PathBuf]) -> Vec<DiscoveredModel> {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return vec![],
    };

    let mut raw: Vec<DiscoveredModel> = Vec::new();

    // Ferrumox own model dir
    let ferrumox_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ferrumox")
        .join("models");
    scan_gguf_dir(&ferrumox_dir, 1, ModelSource::Ferrumox, &mut raw);

    // HuggingFace cache
    let hf_dir = home.join(".cache/huggingface/hub");
    scan_gguf_dir(&hf_dir, 5, ModelSource::HuggingFace, &mut raw);

    // Ollama blobs
    let ollama_dir = home.join(".ollama/models");
    scan_ollama(&ollama_dir, &mut raw);

    // LM Studio
    let lmstudio_dir = home.join(".lmstudio/models");
    scan_gguf_dir(&lmstudio_dir, 4, ModelSource::LmStudio, &mut raw);

    // Custom dirs from --model-dirs
    for dir in extra_dirs {
        scan_gguf_dir(dir, 5, ModelSource::Custom(dir.clone()), &mut raw);
    }

    // Group shards
    let mut models = group_shards(raw);

    // Deduplicate by canonical path
    dedup_by_path(&mut models);

    // Derive human-readable names (skip if already set, e.g. Ollama from manifests)
    for m in &mut models {
        if m.name.is_empty() {
            m.name = derive_name(&m.path, &m.source);
        }
    }

    models.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

    tracing::info!(count = models.len(), "model discovery complete");
    for m in &models {
        tracing::info!(
            name = %m.name,
            source = %m.source,
            size_bytes = m.size_bytes,
            path = %m.path.display(),
            "discovered model"
        );
    }

    models
}

fn scan_gguf_dir(
    dir: &Path,
    max_depth: usize,
    source: ModelSource,
    out: &mut Vec<DiscoveredModel>,
) {
    if !dir.is_dir() {
        return;
    }
    for entry in WalkDir::new(dir)
        .max_depth(max_depth)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str());
        if ext.map(|e| e.eq_ignore_ascii_case("gguf")) != Some(true) {
            continue;
        }
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        out.push(DiscoveredModel {
            name: String::new(),
            path: path.to_path_buf(),
            size_bytes: size,
            source: source.clone(),
            shard_paths: vec![],
        });
    }
}

fn scan_ollama(ollama_dir: &Path, out: &mut Vec<DiscoveredModel>) {
    let blobs_dir = ollama_dir.join("blobs");
    if !blobs_dir.is_dir() {
        return;
    }

    // Build manifest name map: blob hash -> "namespace/model:tag"
    let manifest_names = parse_ollama_manifests(ollama_dir);

    for entry in WalkDir::new(&blobs_dir)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if !has_gguf_magic(path) {
            continue;
        }
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let fname = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();

        // Try to match blob filename to a manifest entry.
        // Ollama blob filenames look like "sha256-<hex>".
        let name = manifest_names.get(fname).cloned().unwrap_or_default();

        out.push(DiscoveredModel {
            name,
            path: path.to_path_buf(),
            size_bytes: size,
            source: ModelSource::Ollama,
            shard_paths: vec![],
        });
    }
}

fn parse_ollama_manifests(ollama_dir: &Path) -> HashMap<String, String> {
    let manifests_dir = ollama_dir.join("manifests");
    let mut map = HashMap::new();
    if !manifests_dir.is_dir() {
        return map;
    }

    // Manifest layout: manifests/registry.ollama.ai/<namespace>/<model>/<tag>
    for entry in WalkDir::new(&manifests_dir)
        .max_depth(5)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let json: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Extract model name from directory structure
        let rel = match path.strip_prefix(&manifests_dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let components: Vec<&str> = rel
            .components()
            .filter_map(|c| c.as_os_str().to_str())
            .collect();

        // Expected: ["registry.ollama.ai", namespace, model, tag]
        let friendly_name = if components.len() >= 4 {
            format!(
                "{}:{}",
                if components[1] == "library" {
                    components[2].to_string()
                } else {
                    format!("{}/{}", components[1], components[2])
                },
                components[3]
            )
        } else {
            continue;
        };

        // Parse layers to find the model blob
        if let Some(layers) = json.get("layers").and_then(|l| l.as_array()) {
            for layer in layers {
                let media_type = layer
                    .get("mediaType")
                    .and_then(|m| m.as_str())
                    .unwrap_or_default();
                if media_type.contains("model") {
                    if let Some(digest) = layer.get("digest").and_then(|d| d.as_str()) {
                        // Digest is "sha256:<hex>", blob filename is "sha256-<hex>"
                        let blob_name = digest.replace(':', "-");
                        map.insert(blob_name, friendly_name.clone());
                    }
                }
            }
        }
    }

    map
}

fn has_gguf_magic(path: &Path) -> bool {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut buf = [0u8; 4];
    match f.read_exact(&mut buf) {
        Ok(()) => buf == GGUF_MAGIC,
        Err(_) => false,
    }
}

/// Detect shard patterns like `model-00001-of-00005.gguf` and group them.
fn group_shards(models: Vec<DiscoveredModel>) -> Vec<DiscoveredModel> {
    let re_shard = regex_shard_pattern();
    let mut shard_groups: HashMap<(String, PathBuf), Vec<DiscoveredModel>> = HashMap::new();
    let mut standalone: Vec<DiscoveredModel> = Vec::new();

    for m in models {
        if let Some(fname) = m.path.file_name().and_then(|n| n.to_str()) {
            if let Some(caps) = re_shard.captures(fname) {
                let prefix = caps.name("prefix").unwrap().as_str().to_string();
                let parent = m.path.parent().unwrap_or(Path::new("")).to_path_buf();
                let key = (prefix, parent);
                shard_groups.entry(key).or_default().push(m);
                continue;
            }
        }
        standalone.push(m);
    }

    for (_, mut shards) in shard_groups {
        shards.sort_by(|a, b| a.path.cmp(&b.path));
        let combined_size: u64 = shards.iter().map(|s| s.size_bytes).sum();
        let first = shards.first().unwrap();
        let shard_paths: Vec<PathBuf> = shards.iter().map(|s| s.path.clone()).collect();
        standalone.push(DiscoveredModel {
            name: first.name.clone(),
            path: first.path.clone(),
            size_bytes: combined_size,
            source: first.source.clone(),
            shard_paths,
        });
    }

    standalone
}

struct ShardRegex;

struct ShardCaptures<'a> {
    prefix: &'a str,
}

struct ShardMatch<'a> {
    text: &'a str,
}

impl<'a> ShardMatch<'a> {
    fn as_str(&self) -> &'a str {
        self.text
    }
}

impl<'a> ShardCaptures<'a> {
    fn name(&self, group: &str) -> Option<ShardMatch<'a>> {
        if group == "prefix" {
            Some(ShardMatch { text: self.prefix })
        } else {
            None
        }
    }
}

impl ShardRegex {
    fn captures<'a>(&self, text: &'a str) -> Option<ShardCaptures<'a>> {
        // Match pattern: <prefix>-NNNNN-of-NNNNN.gguf
        let text_lower = text.to_lowercase();
        if !text_lower.ends_with(".gguf") {
            return None;
        }
        let without_ext = &text[..text.len() - 5]; // strip .gguf
                                                   // Look for -NNNNN-of-NNNNN at the end
                                                   // The digits section can vary in length but must be consistent
        let parts: Vec<&str> = without_ext.rsplitn(4, '-').collect();
        // rsplitn with limit 4 on "prefix-00001-of-00005" gives:
        // ["00005", "of", "00001", "prefix"]
        if parts.len() < 4 {
            return None;
        }
        let total = parts[0];
        let of_part = parts[1];
        let index = parts[2];
        let prefix_text = parts[3];

        if of_part != "of" {
            return None;
        }
        if total.is_empty()
            || index.is_empty()
            || !total.chars().all(|c| c.is_ascii_digit())
            || !index.chars().all(|c| c.is_ascii_digit())
        {
            return None;
        }

        Some(ShardCaptures {
            prefix: prefix_text,
        })
    }
}

fn regex_shard_pattern() -> ShardRegex {
    ShardRegex
}

fn dedup_by_path(models: &mut Vec<DiscoveredModel>) {
    let mut seen = HashSet::new();
    models.retain(|m| {
        let canonical = m.path.canonicalize().unwrap_or_else(|_| m.path.clone());
        seen.insert(canonical)
    });
}

fn derive_name(path: &Path, source: &ModelSource) -> String {
    match source {
        ModelSource::HuggingFace => derive_hf_name(path),
        ModelSource::LmStudio => derive_lmstudio_name(path),
        _ => path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string(),
    }
}

fn derive_hf_name(path: &Path) -> String {
    // Look for "models--org--name" in the path
    for ancestor in path.ancestors() {
        let Some(name) = ancestor.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(rest) = name.strip_prefix("models--") else {
            continue;
        };
        let parts: Vec<&str> = rest.splitn(2, "--").collect();
        if parts.len() == 2 {
            let model_name = format!("{}/{}", parts[0], parts[1]);
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                return format!("{}/{}", model_name, stem);
            }
            return model_name;
        }
    }
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn derive_lmstudio_name(path: &Path) -> String {
    // LM Studio: ~/.lmstudio/models/{org}/{model-file}.gguf
    // Extract the last two components before the filename
    let components: Vec<&str> = path
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();

    // Find "models" in the path and take next component as org
    for (i, c) in components.iter().enumerate() {
        if *c == "models" && i + 2 < components.len() {
            let org = components[i + 1];
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            return format!("{}/{}", org, stem);
        }
    }

    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Parse semicolon-separated directory list into paths.
pub fn parse_model_dirs(input: &str) -> Vec<PathBuf> {
    input
        .split(';')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            let p = PathBuf::from(s);
            crate::cli::expand_tilde(&p)
        })
        .collect()
}

/// Look up a model name in the discovered models list.
/// Returns the path of the first match (exact, then prefix, then contains).
pub fn resolve_discovered(name: &str, models: &[DiscoveredModel]) -> Option<(String, PathBuf)> {
    let lower = name.to_lowercase().replace(':', "-");

    // Exact match on name
    for m in models {
        if m.name.eq_ignore_ascii_case(name) || m.name.to_lowercase().replace(':', "-") == lower {
            return Some((m.name.clone(), m.path.clone()));
        }
    }

    // Name ends with the query (e.g. "llama3" matches "meta-llama/llama3")
    for m in models {
        let m_lower = m.name.to_lowercase().replace(':', "-");
        if m_lower.ends_with(&lower) {
            return Some((m.name.clone(), m.path.clone()));
        }
    }

    // Contains
    for m in models {
        let m_lower = m.name.to_lowercase().replace(':', "-");
        if m_lower.contains(&lower) {
            return Some((m.name.clone(), m.path.clone()));
        }
    }

    // Also try matching against the file stem
    for m in models {
        if let Some(stem) = m.path.file_stem().and_then(|s| s.to_str()) {
            if stem.to_lowercase().contains(&lower) {
                return Some((m.name.clone(), m.path.clone()));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Name derivation ─────────────────────────────────────────────────

    #[test]
    fn test_derive_hf_name() {
        let path = PathBuf::from(
            "/home/user/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/abc123/model.gguf",
        );
        let name = derive_hf_name(&path);
        assert_eq!(name, "meta-llama/Meta-Llama-3.1-8B/model");
    }

    #[test]
    fn test_derive_hf_name_no_pattern() {
        let path = PathBuf::from("/some/random/path/model.gguf");
        let name = derive_hf_name(&path);
        assert_eq!(name, "model");
    }

    #[test]
    fn test_derive_lmstudio_name() {
        let path = PathBuf::from(
            "/home/user/.lmstudio/models/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b.gguf",
        );
        let name = derive_lmstudio_name(&path);
        assert_eq!(name, "TheBloke/mistral-7b");
    }

    #[test]
    fn test_derive_ferrumox_name() {
        let path = PathBuf::from("/home/user/.cache/ferrumox/models/llama-3.2-3b.gguf");
        let name = derive_name(&path, &ModelSource::Ferrumox);
        assert_eq!(name, "llama-3.2-3b");
    }

    #[test]
    fn test_derive_custom_name() {
        let path = PathBuf::from("/data/models/my-model.gguf");
        let name = derive_name(&path, &ModelSource::Custom(PathBuf::from("/data/models")));
        assert_eq!(name, "my-model");
    }

    // ── Shard grouping ──────────────────────────────────────────────────

    #[test]
    fn test_shard_detection() {
        let re = regex_shard_pattern();
        let caps = re.captures("model-00001-of-00005.gguf");
        assert!(caps.is_some());
        assert_eq!(caps.unwrap().name("prefix").unwrap().as_str(), "model");
    }

    #[test]
    fn test_shard_detection_no_match() {
        let re = regex_shard_pattern();
        assert!(re.captures("model.gguf").is_none());
        assert!(re.captures("model-q4.gguf").is_none());
    }

    #[test]
    fn test_group_shards_combines() {
        let models = vec![
            DiscoveredModel {
                name: String::new(),
                path: PathBuf::from("/models/llama-00001-of-00003.gguf"),
                size_bytes: 1000,
                source: ModelSource::Ferrumox,
                shard_paths: vec![],
            },
            DiscoveredModel {
                name: String::new(),
                path: PathBuf::from("/models/llama-00002-of-00003.gguf"),
                size_bytes: 1000,
                source: ModelSource::Ferrumox,
                shard_paths: vec![],
            },
            DiscoveredModel {
                name: String::new(),
                path: PathBuf::from("/models/llama-00003-of-00003.gguf"),
                size_bytes: 1000,
                source: ModelSource::Ferrumox,
                shard_paths: vec![],
            },
            DiscoveredModel {
                name: String::new(),
                path: PathBuf::from("/models/standalone.gguf"),
                size_bytes: 500,
                source: ModelSource::Ferrumox,
                shard_paths: vec![],
            },
        ];

        let grouped = group_shards(models);
        assert_eq!(grouped.len(), 2);

        let sharded = grouped.iter().find(|m| !m.shard_paths.is_empty()).unwrap();
        assert_eq!(sharded.size_bytes, 3000);
        assert_eq!(sharded.shard_paths.len(), 3);

        let single = grouped.iter().find(|m| m.shard_paths.is_empty()).unwrap();
        assert_eq!(single.size_bytes, 500);
    }

    // ── Deduplication ───────────────────────────────────────────────────

    #[test]
    fn test_dedup_by_path() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("model.gguf");
        std::fs::write(&file, b"test").unwrap();

        let mut models = vec![
            DiscoveredModel {
                name: "a".to_string(),
                path: file.clone(),
                size_bytes: 4,
                source: ModelSource::Ferrumox,
                shard_paths: vec![],
            },
            DiscoveredModel {
                name: "b".to_string(),
                path: file.clone(),
                size_bytes: 4,
                source: ModelSource::Custom(tmp.path().to_path_buf()),
                shard_paths: vec![],
            },
        ];

        dedup_by_path(&mut models);
        assert_eq!(models.len(), 1);
    }

    // ── parse_model_dirs ────────────────────────────────────────────────

    #[test]
    fn test_parse_model_dirs_semicolon() {
        let dirs = parse_model_dirs("/path/one;/path/two");
        assert_eq!(dirs.len(), 2);
        assert_eq!(dirs[0], PathBuf::from("/path/one"));
        assert_eq!(dirs[1], PathBuf::from("/path/two"));
    }

    #[test]
    fn test_parse_model_dirs_empty() {
        let dirs = parse_model_dirs("");
        assert!(dirs.is_empty());
    }

    #[test]
    fn test_parse_model_dirs_single() {
        let dirs = parse_model_dirs("/single/path");
        assert_eq!(dirs.len(), 1);
    }

    // ── resolve_discovered ──────────────────────────────────────────────

    #[test]
    fn test_resolve_discovered_exact() {
        let models = vec![DiscoveredModel {
            name: "qwen2.5:7b".to_string(),
            path: PathBuf::from("/test/model.gguf"),
            size_bytes: 100,
            source: ModelSource::Ollama,
            shard_paths: vec![],
        }];
        let result = resolve_discovered("qwen2.5:7b", &models);
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "qwen2.5:7b");
    }

    #[test]
    fn test_resolve_discovered_contains() {
        let models = vec![DiscoveredModel {
            name: "meta-llama/Meta-Llama-3.1-8B/model-q4".to_string(),
            path: PathBuf::from("/test/model.gguf"),
            size_bytes: 100,
            source: ModelSource::HuggingFace,
            shard_paths: vec![],
        }];
        let result = resolve_discovered("llama-3.1", &models);
        assert!(result.is_some());
    }

    #[test]
    fn test_resolve_discovered_not_found() {
        let models = vec![DiscoveredModel {
            name: "llama".to_string(),
            path: PathBuf::from("/test/model.gguf"),
            size_bytes: 100,
            source: ModelSource::Ferrumox,
            shard_paths: vec![],
        }];
        let result = resolve_discovered("mistral", &models);
        assert!(result.is_none());
    }

    // ── has_gguf_magic ──────────────────────────────────────────────────

    #[test]
    fn test_has_gguf_magic_positive() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("blob");
        std::fs::write(&file, b"GGUF\x03\x00\x00\x00").unwrap();
        assert!(has_gguf_magic(&file));
    }

    #[test]
    fn test_has_gguf_magic_negative() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("blob");
        std::fs::write(&file, b"NOT_GGUF").unwrap();
        assert!(!has_gguf_magic(&file));
    }

    #[test]
    fn test_has_gguf_magic_too_short() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("blob");
        std::fs::write(&file, b"GG").unwrap();
        assert!(!has_gguf_magic(&file));
    }

    #[test]
    fn test_has_gguf_magic_missing_file() {
        assert!(!has_gguf_magic(Path::new("/nonexistent/file")));
    }

    // ── ModelSource Display ─────────────────────────────────────────────

    #[test]
    fn test_model_source_display() {
        assert_eq!(format!("{}", ModelSource::Ferrumox), "ferrumox");
        assert_eq!(format!("{}", ModelSource::HuggingFace), "huggingface");
        assert_eq!(format!("{}", ModelSource::Ollama), "ollama");
        assert_eq!(format!("{}", ModelSource::LmStudio), "lmstudio");
        assert_eq!(
            format!("{}", ModelSource::Custom(PathBuf::from("/x"))),
            "custom"
        );
    }

    // ── scan_gguf_dir ───────────────────────────────────────────────────

    #[test]
    fn test_scan_gguf_dir_finds_models() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("model-a.gguf"), b"fake").unwrap();
        std::fs::write(tmp.path().join("model-b.gguf"), b"fake2").unwrap();
        std::fs::write(tmp.path().join("readme.txt"), b"ignore").unwrap();

        let mut out = Vec::new();
        scan_gguf_dir(tmp.path(), 1, ModelSource::Ferrumox, &mut out);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_scan_gguf_dir_nonexistent() {
        let mut out = Vec::new();
        scan_gguf_dir(
            Path::new("/does/not/exist"),
            1,
            ModelSource::Ferrumox,
            &mut out,
        );
        assert!(out.is_empty());
    }

    // ── Ollama name derivation ──────────────────────────────────────────

    #[test]
    fn test_ollama_name_from_manifest_library() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest_dir = tmp
            .path()
            .join("manifests/registry.ollama.ai/library/llama3/latest");
        std::fs::create_dir_all(manifest_dir.parent().unwrap()).unwrap();

        let manifest = serde_json::json!({
            "layers": [{
                "mediaType": "application/vnd.ollama.image.model",
                "digest": "sha256:abc123",
                "size": 1000
            }]
        });
        std::fs::write(&manifest_dir, manifest.to_string()).unwrap();

        let blobs_dir = tmp.path().join("blobs");
        std::fs::create_dir_all(&blobs_dir).unwrap();
        std::fs::write(blobs_dir.join("sha256-abc123"), b"GGUF\x03\x00\x00\x00").unwrap();

        let mut out = Vec::new();
        scan_ollama(tmp.path(), &mut out);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "llama3:latest");
    }
}
