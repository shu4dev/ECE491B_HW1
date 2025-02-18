import os
import time
import argparse
import torch
import numpy as np
import wandb

from model import (
    transformer_lm,
    cross_entropy_loss,
    AdamW,
    get_batch,
    get_lr_cosine_schedule,
    save_checkpoint,
    gradient_clipping,
)

def main(args):
    torch.cuda.empty_cache()
    wandb.init(project=args.wandb_project, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    train_dataset = np.memmap(args.train_file, dtype=np.uint16, mode='r').astype(np.int64)
    val_dataset = np.memmap(args.val_file, dtype=np.uint16, mode='r').astype(np.int64)

    model = transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop
    ).to(device)

    criterion = cross_entropy_loss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps
    )

    total_iters = args.total_iters
    start_time = time.time()

    for iteration in range(1, total_iters + 1):
        model.train()
        x, y = get_batch(train_dataset, args.batch_size, args.context_length, device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        if args.max_norm > 0:
            gradient_clipping(model.parameters(), args.max_norm)
        optimizer.step()
        new_lr = get_lr_cosine_schedule(iteration, args.lr, args.lr_min, args.warmup_iters, total_iters)
        optimizer.set_lr(new_lr)
        if iteration % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"Iteration {iteration}: Train Loss = {loss.item():.4f}, LR = {new_lr:.6f}, Elapsed Time = {elapsed_time:.2f} sec")
            wandb.log({
                "Train/Loss": loss.item(),
                "Train/LR": new_lr,
                "Train/WallclockTime": elapsed_time,
                "iteration": iteration
            }, step=iteration)
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val_dataset, args.batch_size, args.context_length, device)
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val)
                print(f"Iteration {iteration}: Validation Loss = {val_loss.item():.4f}")
                wandb.log({
                    "Validation/Loss": val_loss.item(),
                }, step=iteration)
        if iteration % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            wandb.save(checkpoint_path)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a Transformer LM with experiment tracking using Weights & Biases (wandb).",
        fromfile_prefix_chars='@'
    )
    
    parser.add_argument('--train_file', type=str, default="/home/shu4/ECE491B_HW1/data/Experiment_output/tinystories_train_tokens.npy", required=True, help="Path to training dataset (.npy)")
    parser.add_argument('--val_file', type=str, default="/home/shu4/ECE491B_HW1/data/Experiment_output/tinystories_valid_tokens.npy", required=True, help="Path to validation dataset (.npy)")
    parser.add_argument('--checkpoint_dir', type=str, default="/home/shu4/ECE491B_HW1/data/Experiment_output/tinystories_checkpoints", help="Directory to save checkpoints")
    
    parser.add_argument('--wandb_project', type=str, default="transformer-owt", help="Weights & Biases project name")
    
    parser.add_argument('--vocab_size', type=int, default=50257, help="Vocabulary size")
    parser.add_argument('--context_length', type=int, default=256, help="Context length for training")
    parser.add_argument('--d_model', type=int, default=512, help="Dimension of model embeddings")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of transformer layers")
    parser.add_argument('--num_heads', type=int, default=16, help="Number of attention heads")
    parser.add_argument('--d_ff', type=int, default=2048, help="Dimension of the feed-forward network")
    parser.add_argument('--attn_pdrop', type=float, default=0.1, help="Dropout probability for attention layers")
    parser.add_argument('--residual_pdrop', type=float, default=0.1, help="Dropout probability for residual connections")
    
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--total_iters', type=int, default=10000, help="Total number of training iterations")
    parser.add_argument('--lr', type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument('--lr_min', type=float, default=1e-5, help="Minimum learning rate after decay")
    parser.add_argument('--warmup_iters', type=int, default=1000, help="Iterations for learning rate warmup")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="Weight decay")
    parser.add_argument('--beta1', type=float, default=0.9, help="Beta1 for AdamW")
    parser.add_argument('--beta2', type=float, default=0.999, help="Beta2 for AdamW")
    parser.add_argument('--eps', type=float, default=1e-8, help="Epsilon for AdamW")
    parser.add_argument('--max_norm', type=float, default=1.0, help="Max gradient norm for clipping")
    
    parser.add_argument('--log_interval', type=int, default=100, help="Iterations between logging metrics")
    parser.add_argument('--checkpoint_interval', type=int, default=500, help="Iterations between saving checkpoints")
    
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA training")
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    main(args)