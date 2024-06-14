import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from thop import profile

# Define the LeNet-5 model for MNIST
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(50 * 4 * 4, 800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 50 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_lenet5():
    return LeNet5()

# Dataset preparation (MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Adjust the paths to point to the correct dataset directory
trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)


# Training function for baseline model
def train_baseline_model(net, trainloader, criterion, optimizer, epochs=40):
    net.train()
    for epoch in range(epochs):
        if epoch in [20, 30]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}')
        
# Testing function for the model
def test_model(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

# Save model function
def save_model(net, path):
    torch.save(net.state_dict(), path)

# Load model function
def load_model(net, path):
    state_dict = torch.load(path)
    # Filter out unexpected keys
    state_dict = {k: v for k, v in state_dict.items() if k in net.state_dict()}
    net.load_state_dict(state_dict)

class PruningMethod():
    def prune_filters(self, net, indices):
        conv_layer = 0
        for layer_name, layer_module in net.named_modules():
            if isinstance(layer_module, nn.Conv2d):
                out_channels = indices[conv_layer]
                remaining_indices = list(set(range(layer_module.out_channels)) - set(out_channels))

                layer_module.weight = nn.Parameter(
                    layer_module.weight.data[remaining_indices]
                )

                if layer_module.bias is not None:
                    layer_module.bias = nn.Parameter(
                        layer_module.bias.data[remaining_indices]
                    )

                layer_module.out_channels = len(remaining_indices)
                conv_layer += 1

    def get_indices_topk(self, layer_bounds, prune_ratio):
        indices = int(len(layer_bounds) * prune_ratio)
        k = sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]
        return k

def prune_filters_by_similarity(net, prune_ratio_conv1, prune_ratio_conv2):
    conv_layer = 0
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            filters = module.weight.data.view(module.out_channels, -1)
            cos_sim = F.cosine_similarity(filters.unsqueeze(1), filters.unsqueeze(0), dim=-1).cpu().numpy()
            np.fill_diagonal(cos_sim, 0)

            if conv_layer == 0:
                prune_ratio = prune_ratio_conv1
            else:
                prune_ratio = prune_ratio_conv2

            num_filters_to_prune = max(int(prune_ratio * module.out_channels), 1)
            high_sim_indices = np.argsort(-cos_sim, axis=None)[:num_filters_to_prune]
            high_sim_pairs = np.column_stack(np.unravel_index(high_sim_indices, cos_sim.shape))
            selected_filters = np.unique(high_sim_pairs[:, 1])

            print(f"Module: {module}, Selected Filters: {selected_filters}, Out Channels: {module.out_channels}")

            pruning_method = PruningMethod()
            pruning_method.prune_filters(net, [selected_filters if i == conv_layer else [] for i in range(2)])
            print(f'Pruned {len(selected_filters)} filters from {module}')
            print(f'Number of remaining filters: {module.out_channels - len(selected_filters)}')
            conv_layer += 1
    print(net)

def calculate_similarity(t):
    t = t.view(t.size(0), -1)
    sim_matrix = torch.zeros(t.size(0), t.size(0)).cuda()
    for i in range(t.size(0)):
        for j in range(i + 1, t.size(0)):
            sim_matrix[i, j] = F.cosine_similarity(t[i], t[j], dim=0)
    return sim_matrix

def calculate_flops_and_params(model, input_size):
    input = torch.randn(1, 1, *input_size).cuda()
    flops, params = profile(model, inputs=(input,))
    return flops, params

def calculate_decorrelation_term(sim_matrix, lower_bound=0.3, upper_bound=0.4):
    weak_corr = ((sim_matrix > lower_bound) & (sim_matrix < upper_bound)).float()
    decorrelation_term = weak_corr.sum()
    return decorrelation_term

def iterative_pruning(net, trainloader, testloader, criterion, optimizer, prune_ratio_conv1, prune_ratio_conv2, alpha, beta, epochs=40):
    best_accuracy = 100
    initial_accuracy = test_model(net, testloader)
    best_model_path = 'best_model.pth'
    accuracy = initial_accuracy
    prune_iter = 0

    while abs(best_accuracy - accuracy) < 2:
        print(f'Pruning Iteration {prune_iter + 1}')

        # Load the best model before pruning
        if prune_iter > 0:
            load_model(net, best_model_path)

        prune_filters_by_similarity(net, prune_ratio_conv1, prune_ratio_conv2)

        macs, params = calculate_flops_and_params(net, (1, 28, 28))
        print(f'MACs after pruning: {macs}')
        print(f'Parameters after pruning: {params}')

        for epoch in range(epochs):
            if epoch in [10, 20]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10
            net.train()
            running_loss = 0.0
            old_running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = net(inputs)
                old_loss = criterion(outputs, labels)
                old_running_loss += old_loss.item()

                regularization_term = 0
                decorrelation_term = 0

                for module in net.modules():
                    if isinstance(module, nn.Conv2d):
                        filters = module.weight.data.view(module.out_channels, -1)
                        sim_matrix = calculate_similarity(filters)

                        # Sort similarity values and select the top 2%
                        sim_values = sim_matrix.view(-1)
                        top_2_percent_idx = torch.topk(sim_values, int(0.02 * sim_values.numel()), largest=True).indices
                        regularization_term += torch.exp(-torch.sum(sim_values[top_2_percent_idx]))

                        # Sort weak correlation values and select the top 2%
                        decor_values = ((sim_matrix > 0.3) & (sim_matrix < 0.4)).float().view(-1)
                        top_2_percent_decor_idx = torch.topk(decor_values, int(0.02 * decor_values.numel()), largest=True).indices
                        decorrelation_term += torch.sum(decor_values[top_2_percent_decor_idx])

                new_loss = old_loss + (alpha * regularization_term) - (beta * decorrelation_term)
                new_loss.backward()
                optimizer.step()

                running_loss += new_loss.item()
            print(f'Pruning Iteration {prune_iter + 1}, Epoch [{epoch + 1}/{epochs}], Old Loss: {old_running_loss / len(trainloader):.8f}, New Loss: {running_loss / len(trainloader):.8f}')
            print(f'Regularization term: {regularization_term:.8f}, Decorrelation term: {decorrelation_term:.8f}')

        accuracy = test_model(net, testloader)

        # Save the best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(net, best_model_path)
            print(f'Best model saved with accuracy: {best_accuracy:.2f}%')

        # Stop if the accuracy drop is more than 2%
        if abs(accuracy - best_accuracy) > 2:
            print(f'Pruning stopped as the accuracy drop is less than 2%. Final accuracy: {best_accuracy:.2f}%')
            break

        prune_iter += 1

    # Save the final pruned model
    save_model(net, 'pruned_model.pth')

# Main script
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = get_lenet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

# Load the pre-trained baseline model
load_model(net, 'baseline_model.pth')

# Evaluate baseline model
initial_accuracy = test_model(net, testloader)
baseline_flops, baseline_params = calculate_flops_and_params(net, (28, 28))
print(f'MACs before pruning: {baseline_flops}')
print(f'Parameters before pruning: {baseline_params}')
save_model(net, 'best_model.pth')

# Pruning configurations
prune_ratio_conv1 = 0.04
prune_ratio_conv2 = 0.12
alpha = 0.0001
beta = 0.000001

# Perform iterative pruning with regularization
iterative_pruning(net, trainloader, testloader, criterion, optimizer, prune_ratio_conv1, prune_ratio_conv2, alpha, beta, epochs=40)
