// Entry point: parse CLI subcommand and dispatch.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    ferrumox::cli::run().await
}
