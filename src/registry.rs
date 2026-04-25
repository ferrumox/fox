// Model registry — curated list of popular GGUF models.
//
// The registry is embedded at compile time from `registry.json`.
// It allows `fox pull llama3.2` instead of requiring the full HF repo path.

use std::collections::HashMap;
use std::path::Path;

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

    /// Resolve a user input that may include a quant suffix (`gemma4:q4`,
    /// `gemma4:26b-q4`) to a curated registry entry.
    pub fn resolve_input(&self, input: &str) -> Option<(String, RegistryModel)> {
        self.resolve(registry_lookup_key(input))
    }

    /// Resolve a registry model or alias to the local filename stem expected
    /// after `fox pull` downloads the recommended GGUF file.
    pub fn resolve_file_stem(&self, input: &str) -> Option<String> {
        let (_, model) = self.resolve_input(input)?;
        Path::new(&model.recommended)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(str::to_string)
    }
}

fn registry_lookup_key(input: &str) -> &str {
    if input.contains('/') {
        return input;
    }

    let (name, tag) = match input.split_once(':') {
        Some((name, tag)) => (name, tag),
        None => return input,
    };

    let tag_upper = tag.to_uppercase();
    if tag_upper.starts_with('Q') || tag_upper.starts_with("IQ") || tag_upper.starts_with('F') {
        return name;
    }

    match tag.split_once('-') {
        Some((size, quant)) => {
            let quant_upper = quant.to_uppercase();
            if quant_upper.starts_with('Q')
                || quant_upper.starts_with("IQ")
                || quant_upper.starts_with('F')
            {
                let key_len = name.len() + 1 + size.len();
                &input[..key_len]
            } else {
                input
            }
        }
        None => input,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_gemma4_aliases() {
        let registry = Registry::load();

        let (canonical, model) = registry.resolve("gemma-4").expect("gemma4 alias");
        assert_eq!(canonical, "gemma4");
        assert_eq!(model.repo, "bartowski/google_gemma-4-26B-A4B-it-GGUF");
        assert_eq!(model.recommended, "google_gemma-4-26B-A4B-it-Q4_K_M.gguf");

        let (canonical, _) = registry.resolve("gemma4:26b").expect("gemma4 size alias");
        assert_eq!(canonical, "gemma4");
    }

    #[test]
    fn resolves_registry_inputs_with_quant_suffixes() {
        let registry = Registry::load();

        let (canonical, _) = registry.resolve_input("gemma4:q4").expect("gemma4 q4");
        assert_eq!(canonical, "gemma4");
        let (canonical, _) = registry
            .resolve_input("gemma4:26b-q4")
            .expect("gemma4 26b q4");
        assert_eq!(canonical, "gemma4");
    }

    #[test]
    fn resolves_recommended_file_stem() {
        let registry = Registry::load();
        assert_eq!(
            registry.resolve_file_stem("gemma4").as_deref(),
            Some("google_gemma-4-26B-A4B-it-Q4_K_M")
        );
    }
}
