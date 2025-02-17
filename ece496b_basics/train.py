import os
import time
import argparse
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Import model components from model.py
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
    # Set device (GPU if available and not disabled)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Load datasets using memory-efficient mmap mode
    train_dataset = np.load(args.train_file, mmap_mode='r')
    val_dataset = np.load(args.val_file, mmap_mode='r')

    # Instantiate model with provided hyperparameters
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

    # Set up loss and optimizer
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

    # Set up TensorBoard logging
    writer = SummaryWriter(log_dir=args.log_dir)

    for iteration in range(1, total_iters + 1):
        model.train()
        x, y = get_batch(train_dataset, args.batch_size, args.context_length, device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs.view(-1, args.vocab_size), y.view(-1))
        loss.backward()

        # Apply gradient clipping if enabled.
        if args.max_norm > 0:
            gradient_clipping(model.parameters(), args.max_norm)

        optimizer.step()

        # Update learning rate using cosine schedule.
        new_lr = get_lr_cosine_schedule(iteration, args.lr, args.lr_min, args.warmup_iters, total_iters)
        optimizer.set_lr(new_lr)

        # Logging training metrics every log_interval iterations.
        if iteration % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"Iteration {iteration}: Train Loss = {loss.item():.4f}, LR = {new_lr:.6f}, Elapsed Time = {elapsed_time:.2f} sec")

            writer.add_scalar("Train/Loss", loss.item(), iteration)
            writer.add_scalar("Train/LR", new_lr, iteration)
            writer.add_scalar("Train/WallclockTime", elapsed_time, iteration)

            # Evaluate on validation set.
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val_dataset, args.batch_size, args.context_length, device)
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs.view(-1, args.vocab_size), y_val.view(-1))
                print(f"Iteration {iteration}: Validation Loss = {val_loss.item():.4f}")
                writer.add_scalar("Validation/Loss", val_loss.item(), iteration)

        # Save checkpoints periodically.
        if iteration % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    writer.close()

if __name__ == '__main__':
    # Enable loading arguments from a file with '@'
    parser = argparse.ArgumentParser(
        description="Train a Transformer LM with experiment tracking.",
        fromfile_prefix_chars='@'
    )
    
    # Data paths and checkpoint/log directories.
    parser.add_argument('--train_file', type=str, required=True, help="Path to training dataset (.npy)")
    parser.add_argument('--val_file', type=str, required=True, help="Path to validation dataset (.npy)")
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Directory for TensorBoard logs")
    
    # Model hyperparameters.
    parser.add_argument('--vocab_size', type=int, default=50000, help="Vocabulary size")
    parser.add_argument('--context_length', type=int, default=128, help="Context length for training")
    parser.add_argument('--d_model', type=int, default=512, help="Dimension of model embeddings")
    parser.add_argument('--num_layers', type=int, default=6, help="Number of transformer layers")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--d_ff', type=int, default=2048, help="Dimension of the feed-forward network")
    parser.add_argument('--attn_pdrop', type=float, default=0.1, help="Dropout probability for attention layers")
    parser.add_argument('--residual_pdrop', type=float, default=0.1, help="Dropout probability for residual connections")
    
    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--total_iters', type=int, default=10000, help="Total number of training iterations")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--lr_min', type=float, default=1e-5, help="Minimum learning rate after decay")
    parser.add_argument('--warmup_iters', type=int, default=1000, help="Iterations for learning rate warmup")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="Weight decay")
    parser.add_argument('--beta1', type=float, default=0.9, help="Beta1 for AdamW")
    parser.add_argument('--beta2', type=float, default=0.999, help="Beta2 for AdamW")
    parser.add_argument('--eps', type=float, default=1e-8, help="Epsilon for AdamW")
    parser.add_argument('--max_norm', type=float, default=1.0, help="Max gradient norm for clipping")
    
    # Logging and checkpoint intervals.
    parser.add_argument('--log_interval', type=int, default=100, help="Iterations between logging metrics")
    parser.add_argument('--checkpoint_interval', type=int, default=500, help="Iterations between saving checkpoints")
    
    # Other options.
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA training")
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args)