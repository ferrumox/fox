// Shared visual styling layer for the fox CLI.
//
// All colour logic lives here. When `color_enabled()` returns false every
// helper falls back to plain text so the output is pipe-friendly.

use std::io::{IsTerminal as _, Write};

use crossterm::{
    execute,
    style::{Attribute, Color, Print, SetAttribute, SetForegroundColor},
};

/// Returns `true` when it is safe to emit ANSI colour sequences.
pub fn color_enabled() -> bool {
    std::env::var_os("NO_COLOR").is_none() && std::io::stderr().is_terminal()
}

// ── Core primitive ──────────────────────────────────────────────────────────

fn write_styled(out: &mut impl Write, color: Option<Color>, bold: bool, dim: bool, text: &str) {
    if color_enabled() {
        if bold {
            let _ = execute!(out, SetAttribute(Attribute::Bold));
        }
        if dim {
            let _ = execute!(out, SetAttribute(Attribute::Dim));
        }
        if let Some(c) = color {
            let _ = execute!(out, SetForegroundColor(c));
        }
        let _ = execute!(out, Print(text), SetAttribute(Attribute::Reset));
    } else {
        let _ = write!(out, "{}", text);
    }
}

// ── Public surface ───────────────────────────────────────────────────────────

/// Write styled text to **stderr** (UI chrome: banners, prompts, role labels).
pub fn eprint_styled(color: Option<Color>, bold: bool, dim: bool, text: &str) {
    write_styled(&mut std::io::stderr(), color, bold, dim, text);
}

/// Write styled text to **stdout** (data: tokens, tables).
pub fn print_styled(color: Option<Color>, bold: bool, dim: bool, text: &str) {
    write_styled(&mut std::io::stdout(), color, bold, dim, text);
}

// ── Semantic helpers ─────────────────────────────────────────────────────────

/// Print the REPL banner to stderr:
/// ```
///   🦊  <model_name bold white>
///   ─────────────────────────────── (dim)
///   /bye o Ctrl+D para salir · N tokens  (dim)
/// ```
pub fn print_banner(model_name: &str, context_len: u32) {
    eprint_styled(None, false, false, "  🦊  ");
    eprint_styled(Some(Color::White), true, false, model_name);
    eprintln!();
    eprint_styled(None, false, true, &format!("  {}\n", "─".repeat(44)));
    eprint_styled(
        None,
        false,
        true,
        &format!("  /bye o Ctrl+D para salir · {} tokens\n\n", context_len),
    );
}

/// Print the user-prompt glyph `  ❯ ` (bold cyan) to stderr and flush.
pub fn print_prompt_glyph() {
    eprint_styled(Some(Color::Cyan), true, false, "  ❯ ");
    let _ = std::io::stderr().flush();
}

/// Print `  Fox  ` (bold yellow) to stderr — emitted once before the first
/// response token so the role label appears inline with the streamed text.
pub fn print_fox_label() {
    eprint_styled(Some(Color::Yellow), true, false, "  Fox  ");
}

/// Print `  ✓  <text>` (bold green) to stderr, followed by a newline.
pub fn print_success(text: &str) {
    eprint_styled(Some(Color::Green), true, false, &format!("  ✓  {}", text));
    eprintln!();
}

/// Print the serve-ready banner (green) to stderr:
/// `  🦊  <model>  ·  listening on <addr>`
pub fn print_serve_ready(model: &str, addr: &str) {
    eprint_styled(
        Some(Color::Green),
        false,
        false,
        &format!("  🦊  {}  ·  listening on {}\n", model, addr),
    );
}

// ── Table helpers ─────────────────────────────────────────────────────────────

/// Print a table header row to stdout: each column name bold, padded to its width.
pub fn print_table_header(cols: &[(&str, usize)]) {
    for (name, width) in cols {
        print_styled(
            None,
            true,
            false,
            &format!("{:<width$} ", name, width = width),
        );
    }
    println!();
}

/// Print a dim horizontal separator of `width` `─` characters to stdout.
pub fn print_separator(width: usize) {
    print_styled(None, false, true, &"─".repeat(width));
    println!();
}

/// Print a padded STATUS cell to stdout: `"ok"` → bold green, anything else → plain.
pub fn print_status(status: &str, col_width: usize) {
    if status == "ok" {
        print_styled(
            Some(Color::Green),
            true,
            false,
            &format!("{:<width$} ", status, width = col_width),
        );
    } else {
        print!("{:<width$} ", status, width = col_width);
    }
}

/// Print a padded KV-cache usage cell to stdout with semantic colour:
/// `< 50%` → green, `< 80%` → yellow, `≥ 80%` → red.
pub fn print_kv_cache(usage: f32, col_width: usize) {
    let pct = format!("{:.0}%", usage * 100.0);
    let color = if usage < 0.5 {
        Color::Green
    } else if usage < 0.8 {
        Color::Yellow
    } else {
        Color::Red
    };
    print_styled(
        Some(color),
        false,
        false,
        &format!("{:<width$} ", pct, width = col_width),
    );
}

/// Print `  <key bold+dim padded to 14>  <value>` to stdout, followed by a newline.
pub fn print_kv_pair(key: &str, value: &str) {
    print!("  ");
    print_styled(None, true, true, &format!("{:<14}", key));
    println!("  {}", value);
}
