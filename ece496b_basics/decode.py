import torch
import torch.nn.functional as F
from model import transformer_lm

def decode(
    model,
    prompt,             # list or tensor of token ids; shape: (1, t)
    max_new_tokens=100, # maximum number of tokens to generate
    temperature=1.0,    # softmax temperature for sampling
    top_p=0.9,          # nucleus sampling threshold
    end_token_id=None,  # token id for <|endoftext|>; stop when generated
    device="cpu"
):
    checkpoint = torch.load("/home/shu4/ECE491B_HW1/data/Experiment_output/checkpoints/checkpoint_6000.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # Ensure prompt is a tensor on the right device.
    if not torch.is_tensor(prompt):
        prompt = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    else:
        prompt = prompt.to(device)
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)

    generated = prompt

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Run the model to get logits for each token in the sequence.
            logits = model(generated)  # shape: (1, seq_len, vocab_size)
            # Only consider logits for the last token.
            logits = logits[:, -1, :]  # shape: (1, vocab_size)

            # Apply temperature scaling.
            scaled_logits = logits / temperature
            # Compute probabilities.
            probs = F.softmax(scaled_logits, dim=-1)

            # Top-p (nucleus) sampling:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)

            # Create a mask for tokens to remove.
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask one token to the right to always keep at least one token.
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Set probabilities of filtered-out tokens to 0.
            filtered_probs = probs.clone()
            filtered_probs[0, sorted_indices[0][sorted_indices_to_remove[0]]] = 0.0
            # Renormalize the probabilities.
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

            # Sample the next token.
            next_token = torch.multinomial(filtered_probs, num_samples=1)  # shape: (1, 1)
            # Append the sampled token to the generated sequence.
            generated = torch.cat((generated, next_token), dim=1)

            # If the end-of-text token is generated, stop decoding.
            if end_token_id is not None and next_token.item() == end_token_id:
                break

    return generated

if __name__ == "__main__":
    # Load the trained model.
    model = transformer_lm(
        vocab_size=50000,
        context_length=128,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        attn_pdrop=0.1,
        residual_pdrop=0.1
    )
    # Define the prompt.
    prompt = [101, 2054, 2003, 49999]
    # Generate a completion.
    completion = decode(model, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, end_token_id=49999)
    print(completion)