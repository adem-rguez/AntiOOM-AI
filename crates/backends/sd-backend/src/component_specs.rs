//! Static table describing which split-checkpoint *component kinds* each
//! diffusion-model architecture family needs when `sd.exe` is handed a
//! diffusion-transformer-only GGUF (no VAE / no text encoders bundled in).
//!
//! Matching is keyed on filename substrings for the *architecture family*
//! only (e.g. "flux2", "z-image") — never on a specific HF repo name. New
//! families are added here, not as special cases elsewhere.

/// A required split-checkpoint component. The `&'static str` on `Llm` is a
/// fuzzy filename hint (e.g. `"mistral"`, `"qwen3"`) used only to find a
/// plausible candidate file — not an exact filename to match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentKind {
    ClipL,
    ClipG,
    T5xxl,
    Llm(&'static str),
    Vae,
}

impl ComponentKind {
    /// Human-readable name used in error messages.
    pub fn display_name(&self) -> &'static str {
        match self {
            ComponentKind::ClipL => "clip_l text encoder",
            ComponentKind::ClipG => "clip_g text encoder",
            ComponentKind::T5xxl => "t5xxl text encoder",
            ComponentKind::Llm(_) => "llm text encoder",
            ComponentKind::Vae => "vae",
        }
    }

    /// Filename substrings (case-insensitive) that identify a candidate file
    /// as satisfying this component kind.
    pub fn filename_patterns(&self) -> Vec<&'static str> {
        match self {
            ComponentKind::ClipL => vec!["clip_l", "clip-l"],
            ComponentKind::ClipG => vec!["clip_g", "clip-g"],
            ComponentKind::T5xxl => vec!["t5xxl", "t5-xxl"],
            ComponentKind::Llm(hint) => vec![hint],
            ComponentKind::Vae => vec!["vae"],
        }
    }

    /// Shared-components subfolder (relative to the models root) this kind
    /// should be looked for in when it isn't next to the diffusion model.
    pub fn shared_subdir(&self) -> &'static str {
        match self {
            ComponentKind::Vae => "shared/vae",
            _ => "shared/text_encoders",
        }
    }
}

/// Everything `sd.exe` needs to know to run a given architecture family in
/// component (split-checkpoint) mode.
pub struct ComponentSpec {
    /// Family name, used only for logging/error messages.
    pub family: &'static str,
    pub components: &'static [ComponentKind],
    /// `--vae-format` value, if this family needs one specified explicitly.
    pub vae_format: Option<&'static str>,
    /// Extra `--model-args k=v,...` this family needs (e.g. Chroma/Qwen-Image
    /// specific flags). `None` means no `--model-args` flag is passed.
    pub model_args: Option<&'static str>,
    /// Applied to `ImageParams.cfg_scale` only when the caller didn't set one.
    pub default_cfg_scale: Option<f32>,
    /// Applied to `ImageParams.steps` only when the caller didn't set one.
    pub default_steps: Option<u32>,
}

struct FamilyEntry {
    /// Filename substrings (case-insensitive). Order across `FAMILIES`
    /// matters: more specific patterns (e.g. flux2) must be listed before
    /// broader ones that would otherwise shadow them (e.g. flux1's "flux").
    patterns: &'static [&'static str],
    spec: ComponentSpec,
}

static FAMILIES: &[FamilyEntry] = &[
    // Flux.2 — must come before the Flux1 fallback below.
    FamilyEntry {
        patterns: &["flux-2", "flux2", "flux.2"],
        spec: ComponentSpec {
            family: "flux2",
            components: &[ComponentKind::ClipL, ComponentKind::T5xxl, ComponentKind::Llm("mistral"), ComponentKind::Vae],
            vae_format: Some("flux2"),
            model_args: None,
            default_cfg_scale: None,
            default_steps: None,
        },
    },
    // Z-Image (reuses Flux1's VAE format).
    FamilyEntry {
        patterns: &["z-image", "z_image"],
        spec: ComponentSpec {
            family: "z-image",
            components: &[ComponentKind::Llm("qwen3"), ComponentKind::Vae],
            vae_format: Some("flux"),
            model_args: None,
            default_cfg_scale: Some(1.0),
            default_steps: Some(8),
        },
    },
    // Qwen-Image.
    FamilyEntry {
        patterns: &["qwen-image"],
        spec: ComponentSpec {
            family: "qwen-image",
            components: &[ComponentKind::Llm("qwen"), ComponentKind::Vae],
            vae_format: None,
            model_args: None,
            default_cfg_scale: None,
            default_steps: None,
        },
    },
    // SD3 / SD3.5.
    FamilyEntry {
        patterns: &["sd3"],
        spec: ComponentSpec {
            family: "sd3",
            components: &[ComponentKind::ClipL, ComponentKind::ClipG, ComponentKind::T5xxl, ComponentKind::Vae],
            vae_format: Some("sd3"),
            model_args: None,
            default_cfg_scale: None,
            default_steps: None,
        },
    },
    // Chroma (Flux1-derived).
    FamilyEntry {
        patterns: &["chroma"],
        spec: ComponentSpec {
            family: "chroma",
            components: &[ComponentKind::T5xxl, ComponentKind::Vae],
            vae_format: Some("flux"),
            model_args: None,
            default_cfg_scale: None,
            default_steps: None,
        },
    },
    // Flux1 fallback — must come after flux2's patterns above.
    FamilyEntry {
        patterns: &["flux"],
        spec: ComponentSpec {
            family: "flux1",
            components: &[ComponentKind::ClipL, ComponentKind::T5xxl, ComponentKind::Vae],
            vae_format: Some("flux"),
            model_args: None,
            default_cfg_scale: None,
            default_steps: None,
        },
    },
];

/// Match a diffusion-model filename against the architecture family table.
/// Returns `None` if no family pattern is recognized.
pub fn match_component_spec(filename: &str) -> Option<&'static ComponentSpec> {
    let lower = filename.to_ascii_lowercase();
    FAMILIES
        .iter()
        .find(|entry| entry.patterns.iter().any(|p| lower.contains(p)))
        .map(|entry| &entry.spec)
}
