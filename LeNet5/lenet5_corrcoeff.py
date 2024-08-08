import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchsummary import summary
from thop import profile
import csv
#from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter

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
prune_percentage = [0.04, 0.12]
prune_limits = [1, 2]
optim_lr = 0.0001
lamda = 0.01
alpha = 0.0001
beta = 0.01
regularization_prune_percentage = 0.02
decorrelation_lower_bound = 0.35
decorrelation_higher_bound = 0.4

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
        layer3 = F.relu(self.fc1(layer2_p))
        layer4 = F.relu(self.fc2(layer3))
        layer5 = self.fc3(layer4)
        return layer5

class PruningMethod:
    def prune_filters(self, layer_indices):
        conv_layer = 0
        for layer_name, layer_module in self.named_modules():
            if isinstance(layer_module, th.nn.Conv2d):
                if conv_layer == 0:
                    in_channels = [i for i in range(layer_module.weight.shape[1])]
                else:
                    in_channels = layer_indices[conv_layer - 1]

                out_channels = layer_indices[conv_layer]
                layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])))

                if layer_module.bias is not None:
                    layer_module.bias = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))

                layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.numpy()[:, in_channels])).to('cuda'))
                layer_module.in_channels = len(in_channels)
                layer_module.out_channels = len(out_channels)
                
                conv_layer += 1

            if isinstance(layer_module, th.nn.BatchNorm2d):
                out_channels = layer_indices[conv_layer]
                layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda'))
                layer_module.bias = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))
                layer_module.running_mean = th.from_numpy(layer_module.running_mean.cpu().numpy()[out_channels]).to('cuda')
                layer_module.running_var = th.from_numpy(layer_module.running_var.cpu().numpy()[out_channels]).to('cuda')
                layer_module.num_features = len(out_channels)

            if isinstance(layer_module, nn.Linear):
                conv_layer -= 1
                in_channels = layer_indices[conv_layer]
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

# Define optimizer (scheduler is not used)
optimizer = th.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Load pre-trained model if available
checkpoint = th.load('base.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])  # Load scheduler state
epoch_train_acc = checkpoint['train_acc']
epoch_test_acc = checkpoint['test_acc']

def calculate_correlation_coefficient(layer_weights):
    num_filters = layer_weights.size(0)
    flat_filters = layer_weights.view(num_filters, -1)
    sim_matrix = th.zeros(num_filters, num_filters, device=layer_weights.device)
    for i in range(num_filters):
        for j in range(i + 1, num_filters):
            corr_coef = np.corrcoef(flat_filters[i].cpu().numpy(), flat_filters[j].cpu().numpy())[0, 1]
            sim_matrix[i, j] = corr_coef
            sim_matrix[j, i] = corr_coef  # Ensure symmetry
    return sim_matrix

def calculate_decorrelation_term(sim_matrix, lower_bound=0.3, upper_bound=0.4):
    weak_corr = ((sim_matrix > lower_bound) & (sim_matrix < upper_bound)).astype(np.float32)
    decorrelation_term = weak_corr.sum().item()  # Convert to scalar value
    return decorrelation_term

# Initialize TensorBoard writer
#writer = SummaryWriter('runs/pruning_experiment')

# Get convolutional layers
conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]

# Pruning loop
continue_pruning = True
prunes = 0
best_train_acc = epoch_train_acc
best_test_acc = epoch_test_acc

# Create CSV file and write the header
with open('LeNet5_corrcoeff_summary.csv', mode='w', newline='') as file:
    writer_csv = csv.writer(file)
    writer_csv.writerow(['Iteration', 'Epoch', 'Loss with Regularization', 'CrossEntropy Loss', 'Regularization Term', 'Decorrelation Term', 
                     'Accuracy', 'Conv1', 'Conv2', 'Trainable Parameters', 'Total Flops'])

while continue_pruning:
    # Load the best model from the previous iteration
    if prunes > 0:
        checkpoint = th.load('best_model_new.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])  # Load scheduler state
        epoch_train_acc = checkpoint['train_acc']
        epoch_test_acc = checkpoint['test_acc']

    layer_similarities = []
    selected_indices = [[] for _ in range(len(conv_layers))]
    remaining_indices = [[] for _ in range(len(conv_layers))]
    
    for i, layer in enumerate(conv_layers):
        with th.no_grad():
            filters = layer.weight.data.clone()
            num_filters = filters.size(0)
            
            # Calculate correlation coefficient
            similarity_matrix = []
            for j in range(num_filters):
                for k in range(j + 1, num_filters):
                    corr_coef = np.corrcoef(filters[j].cpu().numpy().flatten(), filters[k].cpu().numpy().flatten())[0, 1]
                    similarity_matrix.append((j, k, corr_coef))
            
            # Sort by correlation coefficient
            similarity_matrix.sort(key=lambda x: x[2], reverse=True)
    
            num_to_prune = max(1, int(prune_percentage[i] * num_filters))  # Ensure at least 1 filter remains
            # Select top filters to prune based on prune_percentage and L1 norm            
            layer_selected_indices = set()
            
            for idx, (filter1, filter2, _) in enumerate(similarity_matrix):
                if idx == 0 or (filter1 not in layer_selected_indices and filter2 not in layer_selected_indices):
                    l1_norm_filter1 = th.norm(layer.weight[filter1], p=1).item()
                    l1_norm_filter2 = th.norm(layer.weight[filter2], p=1).item()
                    
                    # Choose the filter with lower L1 norm
                    if l1_norm_filter1 < l1_norm_filter2:
                        layer_selected_indices.add(filter1)
                    else:
                        layer_selected_indices.add(filter2)

                    if len(layer_selected_indices) >= num_to_prune:
                        break

            selected_indices[i] = list(layer_selected_indices)
            
            # Calculate remaining indices
            layer_remaining_indices = [idx for idx in range(num_filters) if idx not in selected_indices[i]]
            remaining_indices[i] = layer_remaining_indices
    
    print("Selected indices:", selected_indices)
    print("Remaining indices:", remaining_indices)
    
    csv_data = []
    for epoch in range(epochs):
        train_acc = []
        best_train_acc, best_test_acc = 0, 0
        old_running_loss, running_loss = 0, 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Calculate regularization terms
            regularization_term, decorrelation_term = 0, 0
            
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    filters = module.weight.data.reshape(module.out_channels, -1)
                    sim_matrix = calculate_correlation_coefficient(filters)
                    #sim_matrix = calculate_cosine_similarity(filters)

                    # Sort similarity values and select the top 2%
                    sim_values = sim_matrix.view(-1)
                    top_2_percent_idx = th.topk(sim_values, int(regularization_prune_percentage * sim_values.numel()), largest=True).indices
                    regularization_term += th.exp(-th.sum(sim_values[top_2_percent_idx]))

                    # Calculate decorrelation term
                    np_sim_matrix = sim_matrix.cpu().numpy()
                    decorrelation_value = calculate_decorrelation_term(np_sim_matrix, decorrelation_lower_bound, decorrelation_higher_bound)
                    decorrelation_term += th.exp(-th.tensor(decorrelation_value, device=device))  # Convert to tensor

            output = model(inputs)
            old_loss = criterion(output, targets)
            old_running_loss += old_loss.item()
            
            new_loss = old_loss + (alpha * regularization_term) - (beta * decorrelation_term)
            new_loss.backward()
            optimizer.step()
            running_loss += new_loss.item()

            with th.no_grad():
                y_hat = th.argmax(output, 1)
                train_acc.append((y_hat == targets).sum().item())

        epoch_train_acc = sum(train_acc) * 100 / len(trainloader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Train Accuracy: {epoch_train_acc:.2f}%')
        print(f'Pruning Iteration {prunes + 1}, Epoch [{epoch + 1}/{epochs}], Old Loss: {old_running_loss / len(trainloader):.8f}, New Loss: {running_loss / len(trainloader):.8f}')
        print(f'Regularization term: {regularization_term}, Decorrelation term: {decorrelation_term}')

        # Log metrics to TensorBoard
        #writer.add_scalar('Loss/Train', running_loss / len(trainloader), epoch)
        #writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        #writer.add_scalar('Regularization Term', regularization_term, epoch)
        #writer.add_scalar('Decorrelation Term', decorrelation_term, epoch)

        test_acc = []
        
        with th.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                y_hat = th.argmax(output, 1)
                test_acc.append((y_hat == targets).sum().item())
        epoch_test_acc = sum(test_acc) * 100 / len(testloader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Test Accuracy: {epoch_test_acc:.2f}%')

        # Log test accuracy to TensorBoard
        #writer.add_scalar('Accuracy/Test', epoch_test_acc, epoch)

        if epoch_test_acc > best_test_acc:
            best_train_acc = epoch_train_acc
            best_test_acc = epoch_test_acc
            best_model_wts = model.state_dict()
            best_opt_wts = optimizer.state_dict()
            csv_data = [prunes, epoch + 1, (running_loss / len(trainloader)), (old_running_loss / len(trainloader)), regularization_term.item(), 
                        decorrelation_term.item(), best_test_acc]
    
    # Print remaining filters in each convolutional layer
    for i, layer in enumerate(conv_layers):
        print(f'Layer {i+1} - Remaining Filters: {layer.out_channels}')
        csv_data.append(layer.out_channels)
        
    # Calculate FLOPs and parameters
    dummy_input = th.randn(1, 1, 28, 28).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    print(f"Total FLOPs: {flops}, Total Params: {params}")
    csv_data.append(params)
    csv_data.append(flops)
    
    print('csv_data:',csv_data)
    # Write data to CSV
    with open('LeNet5_summary.csv', mode='a', newline='') as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(csv_data)
    
    # Prune filters
    model.prune_filters(remaining_indices)
    
    saved_model = 'best_model_new.pth'
    th.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_acc': best_train_acc,
        'test_acc': best_test_acc
    }, saved_model)

    # Check if desired filter counts are reached
    continue_pruning = any(layer.out_channels > prune_limits[i] for i, layer in enumerate(conv_layers))
    prunes += 1

# Save the pruned model
th.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'train_acc': best_train_acc,
    'test_acc': best_test_acc
}, 'pruned_model_new.pth')

print("Pruning completed successfully.")

# Print model summary
summary(model, (1, 28, 28))

# Close TensorBoard writer
#writer.close()