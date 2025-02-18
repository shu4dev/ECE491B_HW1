import torch
import torch.nn.functional as F
from model import transformer_lm, AdamW, load_checkpoint
import tiktoken


def decode(model, prompt, max_tokens=100, temperature=1.0, top_p=1.0, end_token_id=None):

    if isinstance(prompt, torch.Tensor):
        prompt = prompt.tolist()
    generated = prompt.copy()
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(max_tokens):
            input_tensor = torch.tensor([generated], device=device)
            logits = model(input_tensor)          
            next_logits = logits[0, -1, :]           
            scaled_logits = next_logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[1:] = sorted_mask[:-1].clone()
                sorted_mask[0] = 0
                sorted_probs[sorted_mask] = 0.0
                probs = torch.zeros_like(probs)
                probs[sorted_indices] = sorted_probs
                probs = probs / probs.sum()

            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            if end_token_id is not None and next_token == end_token_id:
                break

    return generated

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab 
    context_length = 256
    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 2048
    attn_pdrop = 0.1
    residual_pdrop = 0.1
    model = transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.001,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    checkpoint_path = "/home/shu4/ECE491B_HW1/data/Experiment_output/owt/checkpoint_9500.pt"
    load_iterations = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer)
    print(f"Loaded checkpoint from iteration {load_iterations}.")
    model.eval()
    prompt_text = "Baseball Prospectus director of technology Harry Pavlidis"
    prompt_tokens = tokenizer.encode(prompt_text)
    generated_ids = decode(
        model=model,
        prompt=prompt_tokens,
        max_tokens=200,
        temperature=1.0,
        top_p=1.0,
    )
    generated_text = tokenizer.decode(generated_ids)
    print("Generated text:", generated_text)
