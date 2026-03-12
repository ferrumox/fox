// Entry point: parse CLI subcommand and dispatch.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    ferrum_engine::cli::run().await
}
