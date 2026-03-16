// `fox models` — list curated models from the embedded registry.

use clap::Parser;

use crate::registry::Registry;

#[derive(Parser, Debug)]
pub struct ModelsArgs {}

pub async fn run_models(_args: ModelsArgs) -> anyhow::Result<()> {
    let registry = Registry::load();
    let models = registry.all();

    let name_w = 14usize;
    let size_w = 7usize;
    let tags_w = 26usize;

    println!(
        "{:<name_w$}  {:>size_w$}  {:<tags_w$}  DESCRIPTION",
        "NAME", "SIZE", "TAGS"
    );
    println!("{}", "-".repeat(name_w + size_w + tags_w + 4 + 40));

    for (name, model) in &models {
        let tags = model.tags.join(", ");
        let size = format!("{:.1} GB", model.size_gb);
        println!(
            "{:<name_w$}  {:>size_w$}  {:<tags_w$}  {}",
            name, size, tags, model.description
        );
    }

    println!();
    println!("Pull a model:   fox pull <name>");
    println!("Example:        fox pull llama3.2");

    Ok(())
}
