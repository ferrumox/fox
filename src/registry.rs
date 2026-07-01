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
