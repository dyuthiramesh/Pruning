import random
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F  # Add this line
import csv
from itertools import zip_longest
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
from collections import defaultdict
from torchsummary import summary
from thop import profile
seed =1787
random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)
th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
th.cuda.set_device(0)
N = 1

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
epochs = 1
custom_epochs = 1#15
new_epochs = 1#30
prune_percentage = [0.04] + [0.10]
prune_limits = [1, 2]  # Desired minimum filter counts
optim_lr = 0.0001
lamda = 0.01

th.cuda.set_device(0)
gpu = th.cuda.is_available()
if not gpu:
    print('qqqq')
else:
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = th.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = th.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2) 

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

   
    def best_tetr_acc(self,prunes):

      print("prunes vaues id ",prunes)
      tr_acc=self.train_accuracy[prunes:]
      te_acc=self.test_accuracy[prunes:]
      best_te_acc=max(te_acc)
      indices = [i for i, x in enumerate(te_acc) if x == best_te_acc]
      temp_tr_acc=[]
      for i in indices:
         temp_tr_acc.append(tr_acc[i])
      best_tr_acc=max(temp_tr_acc)
      
      del self.test_accuracy[prunes:]
      del self.train_accuracy[prunes:]
      self.test_accuracy.append(best_te_acc)
      self.train_accuracy.append(best_tr_acc)
      return best_te_acc,best_tr_acc

    def best_tetr_acc(self):

      tr_acc=self.train_accuracy[:]
      te_acc=self.test_accuracy[:]
      best_te_acc=max(te_acc)
      indices = [i for i, x in enumerate(te_acc) if x == best_te_acc]
      temp_tr_acc=[]
      for i in indices:
         temp_tr_acc.append(tr_acc[i])
      best_tr_acc=max(temp_tr_acc)
      
      del self.test_accuracy[prunes:]
      del self.train_accuracy[prunes:]
      self.test_accuracy.append(best_te_acc)
      self.train_accuracy.append(best_tr_acc)
      return best_te_acc,best_tr_acc

    
    def create_folders(self,total_convs):

      main_dir=strftime("/Results/%b%d_%H:%M:%S%p", localtime() )+"_resnet_56/"
      import os
      current_dir =  os.path.abspath(os.path.dirname(__file__))
      par_dir = os.path.abspath(current_dir + "/../")
      parent_dir=par_dir+main_dir
      path2=os.path.join(parent_dir, "layer_file_info")
      os.makedirs(path2)
      return parent_dir

    def get_writerow(self,k):

      s='wr.writerow(['

      for i in range(k):

          s=s+'d['+str(i)+']'

          if(i<k-1):
             s=s+','
          else:
             s=s+'])'

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
    
import torch as th
import torch.nn as nn
import numpy as np

class PruningMethod:
    def prune_filters(self, indices):
        conv_layer = 0
        for layer_name, layer_module in self.named_modules():
            if isinstance(layer_module, th.nn.Conv2d) and 'conv' in layer_name:
                if layer_name == 'conv1':
                    in_channels = list(range(layer_module.weight.shape[1]))
                    out_channels = indices[conv_layer]
                    new_weight = th.nn.Parameter(layer_module.weight.data[out_channels].clone().to('cuda'))
                    layer_module.weight = new_weight
                elif layer_name.startswith('conv2'):
                    in_channels = indices[conv_layer - 1]
                    out_channels = list(range(layer_module.weight.shape[0]))
                    new_weight = th.nn.Parameter(layer_module.weight.data[:, in_channels].clone().to('cuda'))
                    layer_module.weight = new_weight
                    conv_layer += 1
                
                layer_module.in_channels = len(in_channels)
                layer_module.out_channels = len(out_channels)
            
            elif isinstance(layer_module, th.nn.BatchNorm2d) and 'bn' in layer_name:
                out_channels = indices[conv_layer]
                new_weight = th.nn.Parameter(layer_module.weight.data[out_channels].clone().to('cuda'))
                new_bias = th.nn.Parameter(layer_module.bias.data[out_channels].clone().to('cuda'))
                layer_module.weight = new_weight
                layer_module.bias = new_bias
                layer_module.running_mean = layer_module.running_mean.clone().to('cuda')
                layer_module.running_var = layer_module.running_var.clone().to('cuda')
                layer_module.num_features = len(out_channels)
                conv_layer += 1


    def get_indices_topk(self, layer_bounds, layer_num, prune_limit, prune_value):
        i = layer_num
        indices = prune_value[i]

        p = len(layer_bounds)
        if (p - indices) < prune_limit:
            prune_value[i] = p - prune_limit
            indices = prune_value[i]

        k = sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]
        return k

    def get_indices_bottomk(self, layer_bounds, layer_num, prune_limit):
        k = sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[-prune_limit:]
        return k

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResBasicBlock(nn.Module,Network,PruningMethod):
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
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes-inplanes-(planes//4)), "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu2(out)

        return out


class ResNet(nn.Module,Network,PruningMethod):
    def __init__(self, block, num_layers, covcfg,num_classes=10):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6
        self.covcfg = covcfg
        self.num_layers = num_layers

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)


        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(1,block, 16, blocks=n, stride=1)
        self.layer2 = self._make_layer(2,block, 32, blocks=n, stride=2)
        self.layer3 = self._make_layer(3,block, 64, blocks=n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if num_layers == 110:
            self.linear = nn.Linear(64 * block.expansion, num_classes)
        else:
            self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.initialize()
        self.layer_name_num={}
        self.pruned_filters={}
        self.remaining_filters={}

        self.remaining_filters_each_epoch=[]

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,a, block, planes, blocks, stride):
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

        if self.num_layers == 110:
            x = self.linear(x)
        else:
            x = self.fc(x)

        return x


def resnet_56():
    cov_cfg = [(3 * i + 2) for i in range(9 * 3 * 2 + 1)]
    return ResNet(ResBasicBlock, 56, cov_cfg)

class PruningResNet(ResNet, PruningMethod):
    pass

# Load the model
model = resnet_56()

# Define optimizer and scheduler
optimizer = th.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Load pre-trained model if available
checkpoint = th.load('resnet56_base.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])
epoch_train_acc = checkpoint['train_acc']
epoch_test_acc = checkpoint['test_acc']

# Get convolutional layers
conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]

# Helper function to calculate cosine similarity between filters
def calculate_cosine_similarity(layer_weights):
    num_filters = layer_weights.shape[0]
    flat_filters = layer_weights.reshape(num_filters, -1).cpu().numpy()
    similarity_matrix = cosine_similarity(flat_filters)
    return similarity_matrix

# Pruning loop
continue_pruning = True
prunes = 0
best_train_acc = epoch_train_acc
best_test_acc = epoch_test_acc

# Assuming 'inputs' and 'targets' are already on 'device' (cuda or cpu)
for inputs, targets in trainloader:
    inputs, targets = inputs.to(device), targets.to(device)

    # Move entire model to GPU
    model = model.to(device)

    # Forward pass
    output = model(inputs)

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
            if i < len(prune_limits) and len(selected_indices) >= prune_limits[i]:
                break
            row, col = divmod(idx, num_filters)
            if row != col and row not in selected_indices and col not in selected_indices:
                selected_indices.extend([row, col])
            i += 1  # Increment i inside the loop if applicable

        inc_indices.append(selected_indices)

        # Ensure i is within the valid range of prune_limits
        if i < len(prune_limits):
            unimp_indices_layer = model.get_indices_topk(sim_matrix.sum(axis=0).tolist(), min(i, len(prune_limits) - 1), prune_limits[min(i, len(prune_limits) - 1)], prune_percentage)
        else:
            unimp_indices_layer = []
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