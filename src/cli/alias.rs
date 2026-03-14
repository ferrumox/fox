// `fox alias` — manage model name aliases.
//
// fox alias set phi phi-4-mini-instruct-Q4_K_M
// fox alias list
// fox alias rm phi

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use crossterm::style::Color;

use super::theme;

#[derive(Parser, Debug)]
pub struct AliasArgs {
    #[command(subcommand)]
    pub command: AliasCommand,

    /// Path to aliases TOML file (default: ~/.config/ferrumox/aliases.toml)
    #[arg(long, env = "FOX_ALIAS_FILE", global = true)]
    pub alias_file: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum AliasCommand {
    /// Add or update an alias
    Set {
        /// Short alias name
        name: String,
        /// Model stem or full filename to map to
        value: String,
    },
    /// List all defined aliases
    List,
    /// Remove an alias
    Rm {
        /// Alias name to remove
        name: String,
    },
}

fn aliases_path(override_path: Option<&PathBuf>) -> PathBuf {
    if let Some(p) = override_path {
        return p.clone();
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".config/ferrumox/aliases.toml")
}

fn write_aliases(path: &PathBuf, aliases: &HashMap<String, String>) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut content = String::from("[aliases]\n");
    let mut sorted: Vec<(&String, &String)> = aliases.iter().collect();
    sorted.sort_by_key(|(k, _)| k.as_str());
    for (k, v) in sorted {
        content.push_str(&format!("{} = {:?}\n", k, v));
    }
    std::fs::write(path, content)?;
    Ok(())
}

pub async fn run_alias(args: AliasArgs) -> Result<()> {
    let path = aliases_path(args.alias_file.as_ref());
    let mut aliases = super::load_aliases(Some(path.clone()));

    match args.command {
        AliasCommand::Set { name, value } => {
            let is_update = aliases.contains_key(&name);
            aliases.insert(name.clone(), value.clone());
            write_aliases(&path, &aliases)?;
            let action = if is_update { "Updated" } else { "Added" };
            theme::print_success(&format!("{} alias {} → {}", action, name, value));
        }

        AliasCommand::List => {
            if aliases.is_empty() {
                eprintln!(
                    "No aliases defined. Use `fox alias set <name> <model>` to add one."
                );
                return Ok(());
            }
            let max_name = aliases.keys().map(|k| k.len()).max().unwrap_or(4).max(4);
            let max_val = aliases.values().map(|v| v.len()).max().unwrap_or(5).max(5);
            theme::print_table_header(&[("ALIAS", max_name + 2), ("MODEL", max_val + 2)]);
            theme::print_separator(max_name + max_val + 6);
            let mut sorted: Vec<(&String, &String)> = aliases.iter().collect();
            sorted.sort_by_key(|(k, _)| k.as_str());
            for (k, v) in sorted {
                print!("{:<width$} ", k, width = max_name + 2);
                theme::print_styled(Some(Color::Cyan), false, false, v);
                println!();
            }
        }

        AliasCommand::Rm { name } => {
            if aliases.remove(&name).is_none() {
                anyhow::bail!("alias '{}' not found", name);
            }
            write_aliases(&path, &aliases)?;
            theme::print_success(&format!("Removed alias {}", name));
        }
    }

    Ok(())
}
