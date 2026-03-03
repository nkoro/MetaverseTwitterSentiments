"""
Knowledge-Aware Active Learning (KA2L) Semantic Router
A Gradio Web UI demo that uses GPT-2 hidden-state variance as a proxy for
Semantic Entropy to route supply-chain prompts to a local or cloud model.
"""

import torch
import gradio as gr
from transformers import GPT2Tokenizer, GPT2Model

# ---------------------------------------------------------------------------
# 1. Mock Supply Chain Data
# ---------------------------------------------------------------------------
SUPPLY_CHAIN_PROMPTS = {
    "Process invoice #INV-2024-001": "Standard: Process invoice #INV-2024-001",
    "Update shipment tracking for order #ORD-5523": "Standard: Update shipment tracking for order #ORD-5523",
    "Resolve multi-tier tariff dispute between EU supplier and US distributor": (
        "Complex: Resolve multi-tier tariff dispute between EU supplier and US distributor"
    ),
    "Analyse cross-border customs compliance failure with cascading supplier penalties": (
        "Complex: Analyse cross-border customs compliance failure with cascading supplier penalties"
    ),
}

PROMPT_CHOICES = list(SUPPLY_CHAIN_PROMPTS.keys())

# ---------------------------------------------------------------------------
# 2. Load lightweight GPT-2 router model (loaded once at startup)
# ---------------------------------------------------------------------------
_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
_tokenizer.pad_token = _tokenizer.eos_token  # GPT-2 has no pad token by default
_model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
_model.eval()

# Variance threshold separating "known" from "unknown" distributions.
# GPT-2's 768-dim hidden vectors typically show variance in the range 0.3–0.8;
# 0.5 is a balanced mid-point that cleanly separates the two demo categories.
# Adjust or make configurable (e.g. via env var) for production use.
VARIANCE_THRESHOLD = 0.5


def probe_knowledge_distribution(prompt: str) -> torch.Tensor:
    """
    Tokenize *prompt*, pass it through GPT-2 with output_hidden_states=True,
    extract the last token's hidden state from the final layer, and return
    its variance as a scalar tensor (proxy for Semantic Entropy).
    """
    inputs = _tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)

    # outputs.hidden_states is a tuple of (num_layers + 1) tensors,
    # each shaped [batch, seq_len, hidden_size].
    last_layer_hidden = outputs.hidden_states[-1]   # final transformer layer
    last_token_vector = last_layer_hidden[0, -1, :]  # last token, batch=0
    # Variance across the 768 embedding dimensions (unbiased=True, the default).
    # A single-vector element-wise variance is deliberately used here as a cheap
    # scalar proxy for Semantic Entropy — not a cross-sample statistic.
    variance = torch.var(last_token_vector)
    return variance


# ---------------------------------------------------------------------------
# 3. Hybrid Routing Logic
# ---------------------------------------------------------------------------
def route_prompt(prompt: str):
    """
    Run the KA2L probe and decide where to route the prompt.

    Returns
    -------
    telemetry : str   – variance tensor value
    decision  : str   – routing destination and reasoning
    scoring   : str   – placeholder dual-scoring output
    """
    variance = probe_knowledge_distribution(prompt)
    variance_str = str(variance)

    if variance.item() < VARIANCE_THRESHOLD:
        decision = (
            f"Routing Decision: LOCAL OLLAMA (Qwen)\n"
            f"Reason: Low variance ({variance.item():.6f} < {VARIANCE_THRESHOLD}) → "
            "Known Distribution (High Confidence) — cost-efficient local inference."
        )
    else:
        decision = (
            f"Routing Decision: CLOUD API (GPT-4o)\n"
            f"Reason: High variance ({variance.item():.6f} ≥ {VARIANCE_THRESHOLD}) → "
            "Unknown Distribution (High Entropy) — heavy cloud reasoning required."
        )

    scoring = jobbrex_dual_scoring(decision)
    return variance_str, decision, scoring


# ---------------------------------------------------------------------------
# 4. Placeholder Dual-Scoring Function
# ---------------------------------------------------------------------------
def jobbrex_dual_scoring(output: str) -> str:
    """Placeholder for Tier 3 Dual-Scoring integration."""
    return "Scoring Module Offline - Reserved for Tier 3 Dual-Scoring integration."


# ---------------------------------------------------------------------------
# 5. Gradio Blocks UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="KA2L Hybrid Semantic Router") as demo:
    gr.Markdown(
        """
        # Knowledge-Aware Active Learning (KA2L) Semantic Router
        Select a supply-chain prompt. GPT-2 hidden-state **variance** is used as a
        proxy for *Semantic Entropy* to decide whether to route to a cost-efficient
        **local Ollama (Qwen)** instance or the heavy **Cloud GPT-4o** API.
        """
    )

    with gr.Row():
        prompt_dropdown = gr.Dropdown(
            choices=PROMPT_CHOICES,
            label="Supply-Chain Prompt",
            value=PROMPT_CHOICES[0],
        )
        run_btn = gr.Button("Route Prompt", variant="primary")

    with gr.Row():
        telemetry_box = gr.Textbox(
            label="Telemetry – Variance Tensor (Semantic Entropy Proxy)",
            interactive=False,
        )
    with gr.Row():
        decision_box = gr.Textbox(
            label="Routing Decision – Local Ollama vs Cloud GPT-4o",
            interactive=False,
            lines=4,
        )
    with gr.Row():
        scoring_box = gr.Textbox(
            label="Dual-Scoring Output (Tier 3 Placeholder)",
            interactive=False,
        )

    run_btn.click(
        fn=route_prompt,
        inputs=[prompt_dropdown],
        outputs=[telemetry_box, decision_box, scoring_box],
    )

if __name__ == "__main__":
    # server_name="0.0.0.0" is required for Replit (and similar cloud IDEs) so that
    # the Gradio dev server binds to all interfaces and is reachable via the
    # environment's built-in port forwarding.  share=True still publishes a
    # public Gradio tunnel URL alongside the local one.
    demo.launch(server_name="0.0.0.0", share=True)
