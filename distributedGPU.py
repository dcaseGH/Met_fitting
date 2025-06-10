import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import torch.profiler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


def setup(rank, world_size):
    # Run with torchrun so you don't need to set the environment variables manually
    torch.distributed.init_process_group("nccl",
                                         init_method="env://",
                                         rank=rank,
                                         world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()

# --- 1. Define the Neural Network Model ---
class SimpleNN(nn.Module):
    """
    A simple feedforward neural network designed to take 2D input data
    and produce 1D output.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# --- 2. Generate Synthetic 2D Input Data ---
def generate_data(num_samples, input_dim):
    """
    Generates synthetic 2D input data (features) and corresponding 1D labels.
    The relationship is a simple non-linear function for demonstration.
    """
    # Generate random input features
    X = torch.randn(num_samples, input_dim) * 10

    # Generate labels based on a simple non-linear function
    # For example, y = sin(x1) + cos(x2) + noise
    y = (torch.sin(X[:, 0]) + torch.cos(X[:, 1]) + 0.1 * torch.randn(num_samples)).unsqueeze(1)
    return X, y


def train_distributed(
    rank, 
    world_size,
    num_samples=10,
    input_dim=2,
    hidden_dim=64,
    output_dim=1,
    learning_rate=0.01,
    epochs=100,
    batch_size=2
    ):

    print("Rank ", rank)
    setup(rank, world_size)
    X_train, y_train = generate_data(num_samples, input_dim)
    train_dataset = TensorDataset(X_train, y_train)

    #make a model
    model = SimpleNN(input_dim, hidden_dim, output_dim)
    model.to(rank)  # Move model to the appropriate device
    ddp_model = DDP(model, device_ids=[rank])
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(train_dataset, sampler=sampler)
    criterion = nn.MSELoss() # Mean Squared Error for regression

    #note here we distribute parameters as well as data
    optimizer = optim.SGD(ddp_model.parameters())
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/multiproc_profile'),
        with_stack=False
    ) as profiler:
        for epoch in range(epochs):
            #model.train() # Set model to training mode
            #total_loss = 0
            sampler.set_epoch(epoch)  
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(rank), target.to(rank)  # Move data to the appropriate device
                optimizer.zero_grad()  # Clear gradients
                output = ddp_model(data)
                loss = criterion(output, target)  # Compute loss
                loss.backward()
                optimizer.step()  # Update weights
                profiler.step()  # Important: call profiler.step() after each batch
            if rank == 0 and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        cleanup()


# --- 3. Main Training Function ---
def train_multi_gpu_nn(
    num_samples=10,
    input_dim=2,
    hidden_dim=64,
    output_dim=1,
    learning_rate=0.01,
    epochs=100,
    batch_size=2
):
    """
    Trains a neural network on synthetic 2D data using multiple GPUs if available.
    """
    # Check for GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using {torch.cuda.device_count()} GPU(s).")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU.")

    # Generate data
    X_train, y_train = generate_data(num_samples, input_dim)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = SimpleNN(input_dim, hidden_dim, output_dim)
    model = DDP(model)  # Wrap the model with DistributedDataParallel for multi-GPU training


    # Wrap the model with DataParallel if multiple GPUs are available
    #if torch.cuda.device_count() > 1:
    #    print(f"Using DataParallel across {torch.cuda.device_count()} GPUs.")
#   #     model = nn.DataParallel(model) # This distributes the model across available GPUs
    

    # Move the model to the primary device (e.g., cuda:0) or CPU
    #model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")
    # Use torch.profiler to monitor performance (optional)  
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     with_stack=True
    # ) as prof:
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/simple_nn_profile'),
        with_stack=True
    ) as profiler:
        for epoch in range(epochs):
            model.train() # Set model to training mode
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Move data to the appropriate device (GPU/CPU)
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward and optimize
                optimizer.zero_grad() # Clear gradients
                loss.backward()       # Compute gradients
                optimizer.step()      # Update weights

                total_loss += loss.item()
                profiler.step() # Important: call profiler.step() after each batch

            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("Training finished.")
    print("Run `tensorboard --logdir=./log` to view profiling results.")

    # --- 4. Evaluate the Model (Optional) ---
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        # Generate some test data
        X_test, y_test = generate_data(100, input_dim)
        X_test, y_test = X_test.to(device), y_test.to(device)

        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
        print(f"\nTest Loss: {test_loss.item():.4f}")

    # You can save the trained model
    # If using DataParallel, save the state_dict of the original module
    # if torch.cuda.device_count() > 1:
    #     torch.save(model.module.state_dict(), "simple_nn_multi_gpu.pth")
    # else:
    #     torch.save(model.state_dict(), "simple_nn.pth")
    # print("Model saved.")

if __name__ == "__main__":
    # Ensure that your environment has PyTorch installed and CUDA drivers if using GPUs.
    # To run this, save it as a .py file and execute: python your_script_name.py
    # If you have multiple GPUs, PyTorch will automatically use them with DataParallel.
#    train_multi_gpu_nn()
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
    # This will spawn multiple processes for each GPU available.
    # Each process will run the train_distributed function with its own rank.
    #train_distributed()