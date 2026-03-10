// Config file support: ~/.config/ferrumox/config.toml
//
// Values from the config file are applied as environment variables *before*
// clap parses CLI arguments. Because clap already reads env vars for every
// flag, this effectively makes the config file a source of defaults that is
// overridden by explicit CLI flags or already-set env vars.

use std::path::PathBuf;

#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct ConfigFile {
    model_path: Option<String>,
    host: Option<String>,
    port: Option<u16>,
    max_context_len: Option<u32>,
    max_models: Option<usize>,
    keep_alive_secs: Option<u64>,
    system_prompt: Option<String>,
    gpu_memory_fraction: Option<f32>,
    max_batch_size: Option<usize>,
    block_size: Option<usize>,
    hf_token: Option<String>,
    alias_file: Option<String>,
    json_logs: Option<bool>,
}

/// Load `~/.config/ferrumox/config.toml` (or `$FOX_CONFIG`) and set any
/// values as environment variables, skipping ones already set.
///
/// Must be called **before** `Cli::parse()`.
pub fn load_config_into_env() {
    let path = config_path();
    if !path.exists() {
        return;
    }

    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("warning: could not read config file {:?}: {}", path, e);
            return;
        }
    };

    let cfg: ConfigFile = match toml::from_str(&content) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("warning: could not parse config file {:?}: {}", path, e);
            return;
        }
    };

    set_if_unset("FOX_MODEL_PATH", cfg.model_path);
    set_if_unset("FOX_HOST", cfg.host);
    set_if_unset("FOX_PORT", cfg.port.map(|v| v.to_string()));
    set_if_unset("FOX_MAX_CONTEXT_LEN", cfg.max_context_len.map(|v| v.to_string()));
    set_if_unset("FOX_MAX_MODELS", cfg.max_models.map(|v| v.to_string()));
    set_if_unset("FOX_KEEP_ALIVE_SECS", cfg.keep_alive_secs.map(|v| v.to_string()));
    set_if_unset("FOX_SYSTEM_PROMPT", cfg.system_prompt);
    set_if_unset("FOX_GPU_MEMORY_FRACTION", cfg.gpu_memory_fraction.map(|v| v.to_string()));
    set_if_unset("FOX_MAX_BATCH_SIZE", cfg.max_batch_size.map(|v| v.to_string()));
    set_if_unset("FOX_BLOCK_SIZE", cfg.block_size.map(|v| v.to_string()));
    set_if_unset("HF_TOKEN", cfg.hf_token);
    set_if_unset("FOX_ALIAS_FILE", cfg.alias_file);
    set_if_unset("FOX_JSON_LOGS", cfg.json_logs.map(|v| v.to_string()));
}

fn set_if_unset(var: &str, value: Option<String>) {
    if let Some(v) = value {
        if std::env::var(var).is_err() {
            std::env::set_var(var, v);
        }
    }
}

fn config_path() -> PathBuf {
    if let Ok(p) = std::env::var("FOX_CONFIG") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".config/ferrumox/config.toml")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn write_temp_config(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_config_parse_all_fields() {
        let toml = r#"
host = "127.0.0.1"
port = 9090
max_models = 4
keep_alive_secs = 600
system_prompt = "You are helpful."
gpu_memory_fraction = 0.75
max_batch_size = 16
block_size = 32
max_context_len = 8192
"#;
        let cfg: ConfigFile = toml::from_str(toml).unwrap();
        assert_eq!(cfg.host.as_deref(), Some("127.0.0.1"));
        assert_eq!(cfg.port, Some(9090));
        assert_eq!(cfg.max_models, Some(4));
        assert_eq!(cfg.keep_alive_secs, Some(600));
        assert!((cfg.gpu_memory_fraction.unwrap() - 0.75).abs() < f32::EPSILON);
        assert_eq!(cfg.max_context_len, Some(8192));
    }

    #[test]
    fn test_config_empty_file_is_ok() {
        let cfg: ConfigFile = toml::from_str("").unwrap();
        assert!(cfg.host.is_none());
        assert!(cfg.port.is_none());
    }

    #[test]
    fn test_set_if_unset_does_not_override() {
        // Use a unique env var name to avoid cross-test pollution
        let key = "FOX_TEST_SET_IF_UNSET_NO_OVERRIDE";
        std::env::set_var(key, "original");
        set_if_unset(key, Some("new_value".to_string()));
        assert_eq!(std::env::var(key).unwrap(), "original");
        std::env::remove_var(key);
    }

    #[test]
    fn test_set_if_unset_sets_when_absent() {
        let key = "FOX_TEST_SET_IF_UNSET_ABSENT";
        std::env::remove_var(key);
        set_if_unset(key, Some("hello".to_string()));
        assert_eq!(std::env::var(key).unwrap(), "hello");
        std::env::remove_var(key);
    }

    #[test]
    fn test_set_if_unset_none_leaves_var_absent() {
        let key = "FOX_TEST_SET_IF_UNSET_NONE";
        std::env::remove_var(key);
        set_if_unset(key, None);
        assert!(std::env::var(key).is_err());
    }

    #[test]
    fn test_load_config_into_env_from_file() {
        let toml = "host = \"192.168.1.1\"\nport = 7777\n";
        let f = write_temp_config(toml);

        let host_key = "FOX_TEST_HOST_FROM_FILE";
        let port_key = "FOX_TEST_PORT_FROM_FILE";

        // Simulate what load_config_into_env does for two fields
        let cfg: ConfigFile = toml::from_str(toml).unwrap();
        std::env::remove_var(host_key);
        std::env::remove_var(port_key);
        set_if_unset(host_key, cfg.host);
        set_if_unset(port_key, cfg.port.map(|p| p.to_string()));

        assert_eq!(std::env::var(host_key).unwrap(), "192.168.1.1");
        assert_eq!(std::env::var(port_key).unwrap(), "7777");

        std::env::remove_var(host_key);
        std::env::remove_var(port_key);
        drop(f);
    }
}
