import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F  # Add this line
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
from collections import defaultdict
from torchsummary import summary
from thop import profile

# Initialize random seed for reproducibility
seed = 1787
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

# Set device
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Parameters
epochs = 40
custom_epochs = 15
new_epochs = 30
prune_percentage = [0.04] + [0.10]
prune_limits = [1, 2]  # Desired minimum filter counts
optim_lr = 0.0001
lamda = 0.01

# Data loaders
trainloader = th.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', download=True, train=True,
                               transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=100, shuffle=True)

testloader = th.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', download=True, train=False,
                               transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=100, shuffle=True)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 4 * 4, 800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, 10)
        self.a_type='relu'
        for m in self.modules():
            self.weight_init(m)
        self.softmax = nn.Softmax(dim=1)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=self.a_type)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        layer1 = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        layer2 = F.max_pool2d(F.relu(self.conv2(layer1)), 2)
        layer2_p = layer2.view(-1, int(layer2.nelement() / layer2.shape[0]))
        layer3= F.relu(self.fc1(layer2_p))
        layer4 = F.relu(self.fc2(layer3))
        layer5 = self.fc3(layer4)
        return layer5

class PruningMethod:
    def prune_filters(self, indices):
        conv_layer = 0
        for layer_name, layer_module in self.named_modules():
            if isinstance(layer_module, th.nn.Conv2d):
                if conv_layer == 0:
                    in_channels = [i for i in range(layer_module.weight.shape[1])]
                else:
                    in_channels = indices[conv_layer - 1]

                out_channels = indices[conv_layer]
                layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])))

                if layer_module.bias is not None:
                    layer_module.bias = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))

                layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.numpy()[:, in_channels])).to('cuda'))

                layer_module.in_channels = len(in_channels)
                layer_module.out_channels = len(out_channels)
                conv_layer += 1

            if isinstance(layer_module, th.nn.BatchNorm2d):
                out_channels = indices[conv_layer]
                layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda'))
                layer_module.bias = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))
                layer_module.running_mean = th.from_numpy(layer_module.running_mean.cpu().numpy()[out_channels]).to('cuda')
                layer_module.running_var = th.from_numpy(layer_module.running_var.cpu().numpy()[out_channels]).to('cuda')
                layer_module.num_features = len(out_channels)

            if isinstance(layer_module, nn.Linear):
                conv_layer -= 1
                in_channels = indices[conv_layer]
                weight_linear = layer_module.weight.data.cpu().numpy()
                size = 4 * 4
                expanded_in_channels = []
                for i in in_channels:
                    for j in range(size):
                        expanded_in_channels.extend([i * size + j])
                layer_module.weight = th.nn.Parameter(th.from_numpy(weight_linear[:, expanded_in_channels]).to('cuda'))
                layer_module.in_features = len(expanded_in_channels)
                break

    def get_indices_topk(self, layer_bounds, i, prune_limit, prune_percentage):
        indices = int(len(layer_bounds) * prune_percentage[i]) + 1
        p = len(layer_bounds)
        if (p - indices) < prune_limit:
            remaining = p - prune_limit
            indices = remaining
        k = sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]
        return k

    def get_indices_bottomk(self, layer_bounds, i, prune_limit):
        k = sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[-prune_limit:]
        return k

class PruningLeNet(LeNet, PruningMethod):
    pass

# Load the model
model = PruningLeNet().to(device)

# Define optimizer and scheduler
optimizer = th.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Load pre-trained model if available
checkpoint = th.load('base.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])
epoch_train_acc = checkpoint['train_acc']
epoch_test_acc = checkpoint['test_acc']

# Helper function to calculate cosine similarity between filters
def calculate_cosine_similarity(layer_weights):
    num_filters = layer_weights.shape[0]
    flat_filters = layer_weights.reshape(num_filters, -1).cpu().numpy()
    similarity_matrix = cosine_similarity(flat_filters)
    return similarity_matrix

# Get convolutional layers
conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]

# Pruning loop
continue_pruning = True
prunes = 0
best_train_acc = epoch_train_acc
best_test_acc = epoch_test_acc

while continue_pruning:
    # Calculate cosine similarity for each layer
    layer_similarities = []
    for layer in conv_layers:
        with th.no_grad():
            similarity_matrix = calculate_cosine_similarity(layer.weight)
            layer_similarities.append(similarity_matrix)

    # Select filters to prune based on cosine similarity
    inc_indices = []
    unimp_indices = []
    dec_indices = []
    remaining_indices = []

    for i, sim_matrix in enumerate(layer_similarities):
        num_filters = sim_matrix.shape[0]
        sim_flat = sim_matrix.flatten()
        sorted_indices = np.argsort(sim_flat)[::-1]
        selected_indices = []
        for idx in sorted_indices:
            if len(selected_indices) >= prune_limits[i]:
                break
            row, col = divmod(idx, num_filters)
            if row != col and row not in selected_indices and col not in selected_indices:
                selected_indices.extend([row, col])

        inc_indices.append(selected_indices)
        unimp_indices_layer = model.get_indices_topk(sim_matrix.sum(axis=0).tolist(), i, prune_limits[i], prune_percentage)
        unimp_indices.append(unimp_indices_layer)
        dec_indices.append(list(set(selected_indices + unimp_indices_layer)))
        remaining_indices.append([j for j in range(num_filters) if j not in unimp_indices_layer])

    # Custom regularization
    optimizer = th.optim.SGD(model.parameters(), lr=optim_lr, momentum=0.9)
    for epoch in range(custom_epochs):
        train_acc = []
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Calculate regularization terms
            reg = th.zeros(1).to(device)
            for i, layer in enumerate(conv_layers):
                dec_weight = sum(layer.weight[idx].norm(1) for idx in dec_indices[i])
                inc_weight = sum(layer.weight[idx].norm(1) for idx in inc_indices[i])
                reg += lamda * (dec_weight - inc_weight)

            output = model(inputs)
            loss = criterion(output, targets) + reg
            loss.backward()
            optimizer.step()

            with th.no_grad():
                y_hat = th.argmax(output, 1)
                train_acc.append((y_hat == targets).sum().item())

        epoch_train_acc = sum(train_acc) * 100 / len(trainloader.dataset)
        print(f'Epoch [{epoch+1}/{custom_epochs}], Train Accuracy: {epoch_train_acc:.2f}%')

        test_acc = []
        with th.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                y_hat = th.argmax(output, 1)
                test_acc.append((y_hat == targets).sum().item())
        epoch_test_acc = sum(test_acc) * 100 / len(testloader.dataset)
        print(f'Epoch [{epoch+1}/{custom_epochs}], Test Accuracy: {epoch_test_acc:.2f}%')

        if epoch_test_acc > best_test_acc:
            best_train_acc = epoch_train_acc
            best_test_acc = epoch_test_acc
            best_model_wts = model.state_dict()
            best_opt_wts = optimizer.state_dict()
            best_sch_wts = scheduler.state_dict()

    # Prune filters
    model.prune_filters(remaining_indices)

    # Print remaining filters in each convolutional layer
    for i, layer in enumerate(conv_layers):
        print(f'Layer {i+1} - Remaining Filters: {layer.out_channels}')

    # Check if desired filter counts are reached
    continue_pruning = any(layer.out_channels > prune_limits[i] for i, layer in enumerate(conv_layers))

    # Fine-tuning
    optimizer = th.optim.SGD(model.parameters(), lr=optim_lr, momentum=0.9)
    scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    for epoch in range(new_epochs):
        train_acc = []
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            with th.no_grad():
                y_hat = th.argmax(output, 1)
                train_acc.append((y_hat == targets).sum().item())

        epoch_train_acc = sum(train_acc) * 100 / len(trainloader.dataset)
        print(f'Epoch [{epoch+1}/{new_epochs}], Train Accuracy: {epoch_train_acc:.2f}%')

        test_acc = []
        with th.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                y_hat = th.argmax(output, 1)
                test_acc.append((y_hat == targets).sum().item())
        epoch_test_acc = sum(test_acc) * 100 / len(testloader.dataset)
        print(f'Epoch [{epoch+1}/{new_epochs}], Test Accuracy: {epoch_test_acc:.2f}%')

        if epoch_test_acc > best_test_acc:
            best_train_acc = epoch_train_acc
            best_test_acc = epoch_test_acc
            best_model_wts = model.state_dict()
            best_opt_wts = optimizer.state_dict()
            best_sch_wts = scheduler.state_dict()

    prunes += 1

# Save the pruned model
th.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'train_acc': best_train_acc,
    'test_acc': best_test_acc
}, 'pruned_model.pth')

print("Pruning completed successfully.")

# Print model summary
summary(model, (1, 28, 28))

# Calculate FLOPs and parameters
dummy_input = th.randn(1, 1, 28, 28).to(device)
flops, params = profile(model, inputs=(dummy_input,))
print(f"Total FLOPs: {flops}, Total Params: {params}")
