import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
from torchsummary import summary
from thop import profile
import os
import csv
import math
import time
from datetime import timedelta

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

custom_epochs = 1
AP_custom_epochs = 1

prune_value=[1,2,4]
#prune_limits=[6]*18*3
prune_limits=[8]*9 + [15]*9 + [30]*9 #Make it 9, 9, 9, figure out what these 2 actually mean
#8, 17, 34s

alpha = 0.001
beta = 0.001
AP_alpha = 10

regularization_prune_percentage = 0.02
decorrelation_lower_bound = 0.3
decorrelation_higher_bound = 0.4


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
trainloader = th.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = th.utils.data.DataLoader(testset, batch_size=100, shuffle=True) 


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

class PruningMethod():
    
    def prune_filters(self, indices):
        conv_layer = 0

        for layer_name, layer_module in self.named_modules():

            if isinstance(layer_module, th.nn.Conv2d) and layer_name != 'conv1':

                if layer_name.find('conv1') != -1:
                    in_channels = [i for i in range(layer_module.weight.shape[1])]
                    out_channels = indices[conv_layer]
                    layer_module.weight = th.nn.Parameter(
                        th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda')
                    )

                if layer_name.find('conv2') != -1:
                    in_channels = indices[conv_layer]
                    out_channels = [i for i in range(layer_module.weight.shape[0])]
                    layer_module.weight = th.nn.Parameter(
                        th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[:, in_channels])).to('cuda')
                    )
                    conv_layer += 1

                layer_module.in_channels = len(in_channels)
                layer_module.out_channels = len(out_channels)

            if isinstance(layer_module, th.nn.BatchNorm2d) and layer_name != 'bn1' and layer_name.find('bn1') != -1:
                out_channels = indices[conv_layer]

                layer_module.weight = th.nn.Parameter(
                    th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda')
                )
                layer_module.bias = th.nn.Parameter(
                    th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda')
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
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

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
        self.stride = stride
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential() #identity function
        
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes-inplanes-(planes//4)), "constant", 0)) #to make dimensions same while adding

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out) #batch norm
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(x) #skip connection/concatenation
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

# Load the model
model = resnet_56().to(device)

criterion = nn.CrossEntropyLoss()

checkpoint = th.load('resnet56_base.pth')
model.load_state_dict(checkpoint['model'])

dummy_input = th.randn(1, 3, 32, 32).to(device)
initial_flops, initial_params = profile(model, inputs=(dummy_input,))

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

# Create CSV file and write the header
with open('ResNet56_summary.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration', 'Epochs', 'Total Loss', 'CE Loss', 'Regu', 'Deco', 'Test Acc',
                     'Stage 1', 'Stage 2', 'Stage 3', 'Params', 'Params %','Flops', 'Flops %',
                     'AP-Epochs', 'AP-Finetuning Term', 'H:M'])

while continue_pruning:

    optimizer_pre_prune = th.optim.AdamW(
        model.parameters(),
        lr=0.001,                
        betas=(0.9, 0.999),      
        weight_decay=2e-4        
    )
    scheduler_pre_prune = th.optim.lr_scheduler.MultiStepLR(optimizer_pre_prune, milestones=[20, 30], gamma=0.1)

    layer_similarities = []
    selected_indices = [[] for _ in range(len(conv_layers))]
    remaining_indices = [[] for _ in range(len(conv_layers))]
    
    t1 = time.time()

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

            if i < 9:
                num_to_prune = prune_value[0]
            elif i < 18:
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

    total_cosine_similarity_time = 0
            
    t2 = time.time()
    
    print("Selected indices list:", selected_indices)
    print("Remaining indices list:", remaining_indices)
    
    for epoch in range(custom_epochs):
        train_acc = []
        old_running_loss, running_loss = 0, 0

        if os.path.isfile('test_model.pth'):
            checkpoint = th.load('test_model.pth')
            model.load_state_dict(checkpoint['model'])
            optimizer_pre_prune.load_state_dict(checkpoint['optimizer'])
            scheduler_pre_prune.load_state_dict(checkpoint['scheduler'])
            best_train_acc = checkpoint['train_acc']
            best_test_acc = checkpoint['test_acc']

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_pre_prune.zero_grad()

            # Calculate regularization terms
            regularization_term, decorrelation_term = 0, 0

            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    filters = module.weight.data.reshape(module.out_channels, -1)
                    
                    tsp1 = time.time()
                    sim_matrix = calculate_cosine_similarity(filters)
                    tsp2 = time.time()
                    
                    total_cosine_similarity_time += tsp2 - tsp1

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
            optimizer_pre_prune.step()
            running_loss += new_loss.item()

            with th.no_grad():
                y_hat = th.argmax(output, 1)
                train_acc.append((y_hat == targets).sum().item())

        epoch_train_acc = sum(train_acc) * 100 / len(trainloader.dataset)
        print(f'Epoch [{epoch+1}/{custom_epochs}], Train Accuracy: {epoch_train_acc:.2f}%')
        print(f'Pruning Iteration {prunes + 1}, Epoch [{epoch + 1}/{custom_epochs}], Old Loss: {old_running_loss / len(trainloader):.8f}, New Loss: {running_loss / len(trainloader):.8f}')
        print(f'Regularization term: {regularization_term}, Decorrelation term: {decorrelation_term}')

        test_acc = []

        with th.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                y_hat = th.argmax(output, 1)
                test_acc.append((y_hat == targets).sum().item())
        epoch_test_acc = sum(test_acc) * 100 / len(testloader.dataset)
        print(f'Epoch [{epoch+1}/{custom_epochs}], Test Accuracy: {epoch_test_acc:.2f}%')

        if(epoch==0 or epoch_test_acc > best_test_acc):
            best_train_acc = epoch_train_acc
            best_test_acc = epoch_test_acc

            th.save({'model': model.state_dict(),
                    'optimizer': optimizer_pre_prune.state_dict(),
                    'scheduler': scheduler_pre_prune.state_dict(),
                    'train_acc': best_train_acc,
                    'test_acc': best_test_acc,
                    'running_loss': running_loss,
                    'old_running_loss': old_running_loss}, 'test_model.pth')
            
    if os.path.isfile('test_model.pth'):
        checkpoint = th.load('test_model.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer_pre_prune.load_state_dict(checkpoint['optimizer'])
        scheduler_pre_prune.load_state_dict(checkpoint['scheduler'])
        best_train_acc = checkpoint['train_acc']
        best_test_acc = checkpoint['test_acc']
        best_running_loss = checkpoint['running_loss']
        best_old_running_loss = checkpoint['old_running_loss']

    csv_data = [prunes+1, custom_epochs, (best_running_loss/len(trainloader)), (best_old_running_loss/len(trainloader)), regularization_term.item(), decorrelation_term.item(), best_test_acc, conv_layers[0].out_channels, conv_layers[9].out_channels, conv_layers[18].out_channels]

    print(f'Stage 1 - Remaining Filters: {conv_layers[0].out_channels}')
    print(f'Stage 2 - Remaining Filters: {conv_layers[9].out_channels}')
    print(f'Stage 3 - Remaining Filters: {conv_layers[18].out_channels}')

    flops, params = profile(model, inputs=(dummy_input,))
    print(f"Total FLOPs: {flops}, Total Params: {params}")
    csv_data += (params, (params/initial_params)*100, flops, (flops/initial_flops)*100)
    
    t3 = time.time()
    
    # Prune filters
    model.prune_filters(remaining_indices)
    
    t4 = time.time()
    
    optimizer_post_prune = th.optim.AdamW(
        model.parameters(),
        lr=0.001,                
        betas=(0.9, 0.999),      
        weight_decay=2e-4)
    
    scheduler_post_prune = th.optim.lr_scheduler.MultiStepLR(optimizer_post_prune, milestones=[20, 30], gamma=0.1)
    
    th.save({'model': model.state_dict(),
            'optimizer': optimizer_post_prune.state_dict(),
            'scheduler': scheduler_post_prune.state_dict(),
            'train_acc': 0,
            'test_acc': 0}, 'test_model.pth')

    print("Starting AP Training")

    for AP_epoch in range(AP_custom_epochs):

        if (os.path.isfile('test_model.pth')):
            checkpoint = th.load('test_model.pth')
            model.load_state_dict(checkpoint['model'])
            optimizer_post_prune.load_state_dict(checkpoint['optimizer'])
            scheduler_post_prune.load_state_dict(checkpoint['scheduler'])
            AP_best_train_acc = checkpoint['train_acc']
            AP_best_test_acc = checkpoint['test_acc']

        AP_train_acc = []
        AP_old_running_loss, AP_running_loss = 0, 0

        for AP_inputs, AP_targets in trainloader:
            AP_inputs, AP_targets = AP_inputs.to(device), AP_targets.to(device)
            optimizer_post_prune.zero_grad()

            finetuning_term = 0

            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
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
            optimizer_post_prune.step()
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

        if(AP_epoch==0 or AP_epoch_test_acc > AP_best_test_acc):
            AP_best_train_acc = AP_epoch_train_acc
            AP_best_test_acc = AP_epoch_test_acc

            th.save({
                'model': model.state_dict(),
                'optimizer': optimizer_post_prune.state_dict(),
                'scheduler': scheduler_post_prune.state_dict(),
                'train_acc': AP_best_train_acc,
                'test_acc': AP_best_test_acc
            }, 'test_model.pth')
            
    t5 = time.time()
    
    elapsed_time = timedelta(seconds=t5-t1)
    total_seconds = elapsed_time.total_seconds()
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    csv_data += (AP_custom_epochs, show_finetuning_term, f"{hours}:{minutes}")
    
    e1 = timedelta(seconds=t2-t1)
    sec1 = e1.total_seconds()
    min1 = (sec1 % 3600) // 60
    h1 = sec1 // 3600
    
    e2 = timedelta(seconds=t3-t2)
    sec2 = e2.total_seconds()
    min2 = (sec2 % 3600) // 60
    h2 = sec2 // 3600
    
    e3 = timedelta(seconds=t4-t3)
    sec3 = e3.total_seconds()
    min3 = (sec3 % 3600) // 60
    h3 = sec3 // 3600
    
    e4 = timedelta(seconds=t5-t4)
    sec4 = e4.total_seconds()
    min4 = (sec4 % 3600) // 60
    h4 = sec4 // 3600
    
    print(sec1, min1, h1)
    print(sec2, min2, h2)
    print(sec3, min3, h3)
    print(sec4, min4, h4)
    print("total_cosine_similarity_time:", total_cosine_similarity_time)

    print("Iteration Completed")

    with open('ResNet56_summary.csv', mode='a', newline='') as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(csv_data)

    continue_pruning = False #any(layer.out_channels > prune_limits[i] for i, layer in enumerate(conv_layers))
    prunes += 1

    print("\n")

print("Experiment completed successfully.")

# Print model summary
summary(model, (3, 32, 32))