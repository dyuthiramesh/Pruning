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
import os
import csv
import math

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

custom_epochs = 40
AP_custom_epochs = 40

prune_value=[1,2,4]
#prune_limits=[6]*18*3
prune_limits=[8]*18 + [15]*18 + [30]*18 #Make it 9, 9, 9, figure out what these 2 actually mean
#8, 17, 34s

alpha = 0.01
beta = 0.001
AP_alpha = 10

regularization_prune_percentage = 0.02
decorrelation_lower_bound = 0.3
decorrelation_higher_bound = 0.4

trainloader = th.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('../data', download=True, train=True,
                               transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=100, shuffle=True)

testloader = th.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('../data', download=True, train=False,
                               transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=100, shuffle=True)

class Network():

    def weight_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if self.a_type == 'relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'leaky_relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'tanh':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'sigmoid':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            else:
                raise
                return NotImplemented


    def one_hot(self, y, gpu):

        try:
            y = th.from_numpy(y)
        except TypeError:
            None

        y_1d = y
        if gpu:
            y_hot = th.zeros((y.size(0), th.max(y).int()+1)).cuda()
        else:
            y_hot = th.zeros((y.size(0), th.max(y).int()+1))

        for i in range(y.size(0)):
            y_hot[i, y_1d[i].int()] = 1

        return y_hot

   
    def best_tetr_acc(self, prunes):
        print("prunes values id ", prunes)
        tr_acc = self.train_accuracy[prunes:]
        te_acc = self.test_accuracy[prunes:]
        best_te_acc = max(te_acc)
        indices = [i for i, x in enumerate(te_acc) if x == best_te_acc]
        temp_tr_acc = []
        for i in indices:
            temp_tr_acc.append(tr_acc[i])
        best_tr_acc = max(temp_tr_acc)

        del self.test_accuracy[prunes:]
        del self.train_accuracy[prunes:]
        self.test_accuracy.append(best_te_acc)
        self.train_accuracy.append(best_tr_acc)
        return best_te_acc, best_tr_acc

    def best_tetr_acc(self):
        tr_acc = self.train_accuracy[:]
        te_acc = self.test_accuracy[:]
        best_te_acc = max(te_acc)
        indices = [i for i, x in enumerate(te_acc) if x == best_te_acc]
        temp_tr_acc = []
        for i in indices:
            temp_tr_acc.append(tr_acc[i])
        best_tr_acc = max(temp_tr_acc)

        del self.test_accuracy[prunes:]
        del self.train_accuracy[prunes:]
        self.test_accuracy.append(best_te_acc)
        self.train_accuracy.append(best_tr_acc)
        return best_te_acc, best_tr_acc

    def create_folders(self, total_convs):
        main_dir = strftime("/Results/%b%d_%H:%M:%S%p", localtime()) + "_resnet_110/"
        import os
        current_dir = os.path.abspath(os.path.dirname(__file__))
        par_dir = os.path.abspath(current_dir + "/../")
        parent_dir = par_dir + main_dir
        path2 = os.path.join(parent_dir, "layer_file_info")
        os.makedirs(path2)
        return parent_dir

    def get_writerow(self, k):
        s = 'wr.writerow(['

        for i in range(k):
            s = s + 'd[' + str(i) + ']'
            if i < k - 1:
                s = s + ','
            else:
                s = s + '])'

        return s


    def get_logger(self,file_path):

        logger = logging.getLogger('gal')
        log_format = '%(asctime)s | %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

        return logger

class PruningMethod():
    
    def prune_filters(self, indices):
        conv_layer = 0

        for layer_name, layer_module in self.named_modules():
            if isinstance(layer_module, th.nn.Conv2d) and layer_name != 'conv1':
                if layer_name.find('conv1') != -1:
                    in_channels = [i for i in range(layer_module.weight.shape[1])]
                    out_channels = indices[conv_layer]
                    layer_module.weight = th.nn.Parameter(
                        th.FloatTensor(
                            th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])
                        ).to('cuda')
                    )

                if layer_name.find('conv2') != -1:
                    in_channels = indices[conv_layer]
                    out_channels = [i for i in range(layer_module.weight.shape[0])]
                    layer_module.weight = th.nn.Parameter(
                        th.FloatTensor(
                            th.from_numpy(layer_module.weight.data.cpu().numpy()[:, in_channels])
                        ).to('cuda')
                    )
                    conv_layer += 1

                layer_module.in_channels = len(in_channels)
                layer_module.out_channels = len(out_channels)

            if isinstance(layer_module, th.nn.BatchNorm2d) and layer_name != 'bn1' and layer_name.find('bn1') != -1:
                out_channels = indices[conv_layer]

                layer_module.weight = th.nn.Parameter(
                    th.FloatTensor(
                        th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])
                    ).to('cuda')
                )
                layer_module.bias = th.nn.Parameter(
                    th.FloatTensor(
                        th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])
                    ).to('cuda')
                )
                
                layer_module.running_mean = th.from_numpy(
                    layer_module.running_mean.cpu().numpy()[out_channels]
                ).to('cuda')
                layer_module.running_var = th.from_numpy(
                    layer_module.running_var.cpu().numpy()[out_channels]
                ).to('cuda')

                layer_module.num_features = len(out_channels)

            if isinstance(layer_module, nn.Linear):
                break

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ResBasicBlock(nn.Module, Network, PruningMethod):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes - inplanes - (planes // 4)), "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu2(out)

        return out

class ResNet(nn.Module, Network, PruningMethod):
    def __init__(self, block, num_layers, covcfg, num_classes=10):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6
        self.covcfg = covcfg
        self.num_layers = num_layers

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(1, block, 16, blocks=n, stride=1)
        self.layer2 = self._make_layer(2, block, 32, blocks=n, stride=2)
        self.layer3 = self._make_layer(3, block, 64, blocks=n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Linear(64 * block.expansion, num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, a, block, planes, blocks, stride):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def resnet_110():
    cov_cfg = [(3 * i + 2) for i in range(9 * 6 * 2 + 1)]
    return ResNet(ResBasicBlock, 110, cov_cfg)

# Load the model
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model = resnet_110().to(device)

# Define optimizer and scheduler
optimizer = th.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4, nesterov=True)
scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Load pre-trained model if available
checkpoint = th.load('base.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])
AP_epoch_train_acc = checkpoint['train_acc']
AP_epoch_test_acc = checkpoint['test_acc']

def calculate_cosine_similarity(layer_weights):
    num_filters = layer_weights.size(0)
    flat_filters = layer_weights.view(num_filters, -1)
    sim_matrix = th.zeros(num_filters, num_filters, device=layer_weights.device)
    for i in range(num_filters):
        for j in range(i + 1, num_filters):
            cosine_sim = F.cosine_similarity(flat_filters[i], flat_filters[j], dim=0)
            sim_matrix[i, j] = cosine_sim
            sim_matrix[j, i] = cosine_sim  # Ensure symmetry
    return sim_matrix


def calculate_decorrelation_term(sim_matrix, lower_bound=0.3, upper_bound=0.4):
    weak_corr = ((sim_matrix > lower_bound) & (sim_matrix < upper_bound)).astype(np.float32)
    decorrelation_term = weak_corr.sum().item()  # Convert to scalar value
    return decorrelation_term

# Get convolutional layers for ResNet56
conv_layers = [module for name, module in model.named_modules() if isinstance(module, th.nn.Conv2d) and name!='conv1' and name.find('conv1')!=-1] #So 27 layers

# Pruning loop
continue_pruning = True
prunes = 0
AP_best_train_acc = AP_epoch_train_acc
AP_best_test_acc = AP_epoch_test_acc

# Create CSV file and write the header
with open('Resnet110_summary.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration', 
                     'Epochs', 'Total Loss', 'CE Loss', 'Regu', 'Deco', 'Test Acc',
                     'Stage 1', 'Stage 2', 'Stage 3', 'Params', 'Flops',
                     'AP-Epochs', 'AP-Finetuning Term'])

try:
    while continue_pruning:
        # Load the best model from the previous iteration
        if prunes > 0 and os.path.isfile('newest_model.pth'):
            checkpoint = th.load('newest_model.pth')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            AP_epoch_train_acc = checkpoint['train_acc']
            AP_epoch_test_acc = checkpoint['test_acc']

        layer_similarities = []
        selected_indices = [[] for _ in range(len(conv_layers))]
        remaining_indices = [[] for _ in range(len(conv_layers))]

        for i, layer in enumerate(conv_layers):
            with th.no_grad():
                filters = layer.weight.data.clone()
                num_filters = filters.size(0)

                # Calculate cosine similarity
                similarity_matrix = []
                for j in range(num_filters):
                    for k in range(j + 1, num_filters):
                        cosine_sim = F.cosine_similarity(filters[j].flatten(), filters[k].flatten(), dim=0)
                        similarity_matrix.append((j, k, cosine_sim.item()))

                # Sort by cosine similarity (descending)
                similarity_matrix.sort(key=lambda x: -x[2])
                layer_similarities.append(similarity_matrix)

                if i < 18:
                    num_to_prune = prune_value[0]
                elif i < 36:
                    num_to_prune = prune_value[1]
                else:
                    num_to_prune = prune_value[2]

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

                remaining_indices[i] = [idx for idx in range(num_filters) if idx not in selected_indices[i]]

        print("Selected indices list:", selected_indices)
        print("Remaining indices list:", remaining_indices)

        for epoch in range(custom_epochs):
            train_acc = []
            old_running_loss, running_loss = 0, 0

            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                # Calculate regularization terms
                regularization_term, decorrelation_term = 0, 0

                for module in model.modules():
                    if isinstance(module, nn.Conv2d):
                        filters = module.weight.data.reshape(module.out_channels, -1)
                        sim_matrix = calculate_cosine_similarity(filters)

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
            print(f'Epoch [{epoch+1}/{custom_epochs}], Train Accuracy: {epoch_train_acc:.2f}%')
            print(f'Pruning Iteration {prunes + 1}, Epoch [{epoch + 1}/{custom_epochs}], Old Loss: {old_running_loss / len(trainloader):.8f}, New Loss: {running_loss / len(trainloader):.8f}')
            print(f'Regularization term: {regularization_term}, Decorrelation term: {decorrelation_term}')

            '''# Log metrics to TensorBoard
            writer.add_scalar('Loss/Train', running_loss / len(trainloader), epoch)
            writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
            writer.add_scalar('Regularization Term', regularization_term, epoch)
            writer.add_scalar('Decorrelation Term', decorrelation_term, epoch)'''

            test_acc = []

            with th.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs)
                    y_hat = th.argmax(output, 1)
                    test_acc.append((y_hat == targets).sum().item())
            epoch_test_acc = sum(test_acc) * 100 / len(testloader.dataset)
            print(f'Epoch [{epoch+1}/{custom_epochs}], Test Accuracy: {epoch_test_acc:.2f}%')

            # Log test accuracy to TensorBoard
            # writer.add_scalar('Accuracy/Test', epoch_test_acc, epoch)

        csv_data = [prunes+1, custom_epochs, (running_loss / len(trainloader)), (old_running_loss / len(trainloader)), regularization_term.item(), decorrelation_term.item(), epoch_test_acc, conv_layers[0].out_channels, conv_layers[18].out_channels, conv_layers[36].out_channels]

        print(f'Stage 1 - Remaining Filters: {conv_layers[0].out_channels}')
        print(f'Stage 2 - Remaining Filters: {conv_layers[18].out_channels}')
        print(f'Stage 3 - Remaining Filters: {conv_layers[36].out_channels}')

        # Calculate FLOPs and parameters
        dummy_input = th.randn(1, 3, 32, 32).to(device)
        flops, params = profile(model, inputs=(dummy_input,))
        print(f"Total FLOPs: {flops}, Total Params: {params}")
        csv_data += (params, flops)

        # Prune filters
        model.prune_filters(remaining_indices)

        #AP = After Pruning

        print("Starting AP Training")

        for AP_epoch in range(AP_custom_epochs): 
            AP_train_acc = []
            AP_old_running_loss, AP_running_loss = 0, 0

            for AP_inputs, AP_targets in trainloader:
                AP_inputs, AP_targets = AP_inputs.to(device), AP_targets.to(device)
                optimizer.zero_grad()

                finetuning_term = 0

                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d) and name!='conv1' and name.find('conv1')!=-1:
                        filters = module.weight
                        l2_norm_value = th.norm(filters, p=2).item()
                        finetuning_term += l2_norm_value
            
                finetuning_term = math.log(finetuning_term)
                show_finetuning_term = math.exp(-finetuning_term)
                
                finetuning_term = th.tensor(finetuning_term, device=device)  # Convert to tensor
                finetuning_term = th.exp(-finetuning_term)

                AP_output = model(AP_inputs)
                AP_old_loss = criterion(AP_output, AP_targets)
                AP_old_running_loss += AP_old_loss.item()

                AP_new_loss = AP_old_loss + (AP_alpha * finetuning_term)
                AP_new_loss.backward()
                optimizer.step()
                AP_running_loss += AP_new_loss.item()

                with th.no_grad():
                    AP_y_hat = th.argmax(AP_output, 1)
                    AP_train_acc.append((AP_y_hat == AP_targets).sum().item())

            AP_epoch_train_acc = sum(AP_train_acc) * 100 / len(trainloader.dataset)

            print(f'AP_Epoch [{AP_epoch+1}/{AP_custom_epochs}], AP Train Accuracy: {AP_epoch_train_acc:.2f}%,  AP Old Loss: {AP_old_running_loss / len(trainloader):.8f}, AP New Loss: {AP_running_loss / len(trainloader):.8f}, AP Finetuning term: {finetuning_term}')

            AP_test_acc = []

            with th.no_grad():
                for AP_inputs, AP_targets in testloader:
                    AP_inputs, AP_targets = AP_inputs.to(device), AP_targets.to(device)
                    AP_output = model(AP_inputs)
                    AP_y_hat = th.argmax(AP_output, 1)

                    AP_test_acc.append((AP_y_hat == AP_targets).sum().item())

            AP_epoch_test_acc = sum(AP_test_acc) * 100 / len(testloader.dataset)
            print(f'AP Epoch [{AP_epoch+1}/{AP_custom_epochs}], Test Accuracy: {AP_epoch_test_acc:.2f}%')

            if AP_epoch_test_acc > AP_best_test_acc:
                AP_best_train_acc = AP_epoch_train_acc
                AP_best_test_acc = AP_epoch_test_acc
                
        th.save({'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_acc': AP_best_train_acc,
        'test_acc': AP_best_test_acc}, 'newest_model.pth')

        csv_data += (AP_custom_epochs, show_finetuning_term)

        print("Ended AP training")

        with open('Resnet110_summary.csv', mode='a', newline='') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow(csv_data)

        continue_pruning = any(layer.out_channels > prune_limits[i] for i, layer in enumerate(conv_layers))
        prunes += 1
        
        print("\n")

finally:
    # Save the pruned model
    th.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_acc': AP_best_train_acc,
        'test_acc': AP_best_test_acc
    }, 'pruned_model0.pth')

    print("Pruning completed successfully.")

    # Print model summary
    summary(model, (3, 32, 32))