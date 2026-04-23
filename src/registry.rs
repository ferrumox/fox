// Model registry — curated list of popular GGUF models.
//
// The registry is embedded at compile time from `registry.json`.
// It allows `fox pull llama3.2` instead of requiring the full HF repo path.

use std::collections::HashMap;

use serde::Deserialize;

const REGISTRY_JSON: &str = include_str!("../registry.json");

#[derive(Deserialize, Clone)]
pub struct RegistryModel {
    pub aliases: Vec<String>,
    pub repo: String,
    pub recommended: String,
    pub description: String,
    pub size_gb: f32,
    pub tags: Vec<String>,
}

#[derive(Deserialize)]
struct RegistryFile {
    models: HashMap<String, RegistryModel>,
}

pub struct Registry {
    models: HashMap<String, RegistryModel>,
}

impl Registry {
    pub fn load() -> Self {
        let file: RegistryFile =
            serde_json::from_str(REGISTRY_JSON).expect("embedded registry.json is invalid JSON");
        Self {
            models: file.models,
        }
    }

    /// All models sorted by name.
    pub fn all(&self) -> Vec<(String, RegistryModel)> {
        let mut v: Vec<(String, RegistryModel)> = self
            .models
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    /// Resolve a short name or alias to `(canonical_name, model)`.
    /// Returns `None` if not found — caller should treat input as a raw HF repo.
    pub fn resolve(&self, name: &str) -> Option<(String, RegistryModel)> {
        // Exact key match
        if let Some(m) = self.models.get(name) {
            return Some((name.to_string(), m.clone()));
        }
        // Alias match
        for (key, model) in &self.models {
            if model.aliases.iter().any(|a| a == name) {
                return Some((key.clone(), model.clone()));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_json_parses() {
        let reg = Registry::load();
        assert!(!reg.models.is_empty());
    }

    #[test]
    fn resolve_exact_key() {
        let reg = Registry::load();
        let (name, model) = reg.resolve("llama3.2").expect("llama3.2 should resolve");
        assert_eq!(name, "llama3.2");
        assert!(!model.repo.is_empty());
        assert!(model.recommended.ends_with(".gguf"));
    }

    #[test]
    fn resolve_alias() {
        let reg = Registry::load();
        let (name, model) = reg.resolve("llama3").expect("llama3 alias should resolve");
        assert_eq!(name, "llama3.2");
        assert_eq!(model.repo, "bartowski/Llama-3.2-3B-Instruct-GGUF");
    }

    #[test]
    fn resolve_embed_alias() {
        let reg = Registry::load();
        let (name, model) = reg.resolve("embed").expect("embed alias should resolve");
        assert_eq!(name, "nomic-embed");
        assert!(model.repo.contains("nomic"));
    }

    #[test]
    fn resolve_unknown_returns_none() {
        let reg = Registry::load();
        assert!(reg.resolve("nonexistent-model-xyz").is_none());
    }

    #[test]
    fn resolve_raw_hf_repo_returns_none() {
        let reg = Registry::load();
        assert!(reg
            .resolve("bartowski/Llama-3.2-3B-Instruct-GGUF")
            .is_none());
    }

    #[test]
    fn all_models_have_required_fields() {
        let reg = Registry::load();
        for (name, model) in reg.all() {
            assert!(!name.is_empty(), "model name should not be empty");
            assert!(!model.repo.is_empty(), "{name} missing repo");
            assert!(
                model.recommended.ends_with(".gguf"),
                "{name} recommended should end with .gguf"
            );
            assert!(model.repo.contains('/'), "{name} repo should be owner/name");
        }
    }
}
