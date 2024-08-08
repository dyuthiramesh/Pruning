import random
import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import CosineSimilarity
from torchsummary import summary
from thop import profile
import csv
import numpy as np
import logging
import csv 
from time import localtime, strftime
import os 
import math
from collections import OrderedDict
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
device = 'cuda' if th.cuda.is_available() else 'cpu'
csv_file_path = 'VGG16_CIFAR10_Summary.csv'

# Parameters
epochs = 1
batch_size_tr = 100
batch_size_te = 100
prune_percentage = [0.02]*2 + [0.04]*2 + [0.05]*3 + [0.10]*6
prune_limits = [40]*16
alpha = 0.0001
beta = 0.01
delta = 0.1 # fine-tuning
regularization_prune_percentage = 0.02
decorrelation_lower_bound = 0.35
decorrelation_higher_bound = 0.4

# Data loaders
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
trainloader = th.utils.data.DataLoader(trainset, batch_size=batch_size_tr, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = th.utils.data.DataLoader(testset, batch_size=batch_size_te, shuffle=True)


class PruningMethod():
    def prune_filters(self,indices):
        conv_layer=0
        for layer_name, layer_module in self.named_modules():
            if (isinstance(layer_module, th.nn.Conv2d)):
                if(conv_layer==0):            
                    in_channels=[i for i in range(layer_module.weight.shape[1])]
                else:
                    in_channels=indices[conv_layer-1]
                
                out_channels=indices[conv_layer]
                layer_module.weight = th.nn.Parameter( th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])))
                #layer_module.weight = layer_module.weight.data.cpu().numpy()[out_channels]
                
                if layer_module.bias is not None:
                    #layer_module.bias = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))
                    layer_module.bias = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))
                    
                layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.numpy()[:,in_channels])).to('cuda'))
                layer_module.in_channels=len(in_channels)
                layer_module.out_channels=len(out_channels)
                
            if (isinstance(layer_module, th.nn.BatchNorm2d)):
                out_channels = indices[conv_layer]
                conv_layer += 1
                
                layer_module.weight =th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda'))
                layer_module.bias=th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))
                
                layer_module.running_mean= th.from_numpy(layer_module.running_mean.cpu().numpy()[out_channels]).to('cuda')
                layer_module.running_var=th.from_numpy(layer_module.running_var.cpu().numpy()[out_channels]).to('cuda')
                
                layer_module.num_features= len(out_channels)

            if isinstance(layer_module, nn.Linear):
                conv_layer-=1
                in_channels=indices[conv_layer]
                weight_linear = layer_module.weight.data.cpu().numpy()
                weight_linear_rearranged = np.transpose(weight_linear, (1, 0))

                size=1*1
                expanded_in_channels=[]
                for i in in_channels:
                    for j in range(size):
                        expanded_in_channels.extend([i*size+j])

                weight_linear_rearranged_pruned = weight_linear_rearranged[expanded_in_channels]
                weight_linear_rearranged_pruned = np.transpose(weight_linear_rearranged_pruned, (1, 0))
                layer_module.weight = th.nn.Parameter(th.from_numpy(weight_linear_rearranged_pruned).to('cuda'))

                layer_module.in_features = len(expanded_in_channels)
                break

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
        main_dir=strftime("/Results/%b%d_%H:%M:%S%p", localtime() )+"_vgg/"
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
            
# Define VGG16 model with PruningMethod
class VGG16(nn.Module,Network,PruningMethod):
    def __init__(self, n_c, a_type):
        super(VGG16, self).__init__()
        self.a_type = a_type
        if a_type == 'relu':
            self.activation = nn.ReLU()
        elif a_type == 'tanh':
            self.activation = nn.Tanh()
        elif a_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif a_type == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            print('Not implemented')
            raise

        # First encoder
        self.layer1 = nn.Sequential(*([nn.Conv2d(3, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64),self.activation]))
        self.layer2 = nn.Sequential(*([nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64),self.activation]))
        # Second encoder
        self.layer3 = nn.Sequential(*([nn.Conv2d(64, 128, kernel_size=3, padding=1),nn.BatchNorm2d(128),self.activation]))
        self.layer4 = nn.Sequential(*([nn.Conv2d(128, 128, kernel_size=3, padding=1),nn.BatchNorm2d(128),self.activation]))
        # Third encoder
        self.layer5 = nn.Sequential(*([nn.Conv2d(128, 256, kernel_size=3, padding=1),nn.BatchNorm2d(256),self.activation]))
        self.layer6 = nn.Sequential(*([nn.Conv2d(256, 256, kernel_size=3, padding=1),nn.BatchNorm2d(256),self.activation]))
        self.layer7 = nn.Sequential(*([nn.Conv2d(256, 256, kernel_size=3, padding=1),nn.BatchNorm2d(256),self.activation]))
        # Fourth encoder
        self.layer8 = nn.Sequential(*([nn.Conv2d(256, 512, kernel_size=3, padding=1),nn.BatchNorm2d(512),self.activation]))
        self.layer9 = nn.Sequential(*([nn.Conv2d(512, 512, kernel_size=3, padding=1),nn.BatchNorm2d(512),self.activation]))
        self.layer10 = nn.Sequential(*([nn.Conv2d(512, 512, kernel_size=3, padding=1),nn.BatchNorm2d(512),self.activation]))
        # Fifth encoder
        self.layer11 = nn.Sequential(*([nn.Conv2d(512, 512, kernel_size=3, padding=1),nn.BatchNorm2d(512),self.activation]))
        self.layer12 = nn.Sequential(*([nn.Conv2d(512, 512, kernel_size=3, padding=1),nn.BatchNorm2d(512),self.activation]))
        self.layer13 = nn.Sequential(*([nn.Conv2d(512, 512, kernel_size=3, padding=1),nn.BatchNorm2d(512),self.activation]))
        # Classifier
        self.fc1 = nn.Sequential(*([nn.Linear(512, 512),nn.BatchNorm1d(512),self.activation]))
        self.classifier = nn.Sequential(*([nn.Linear(512, n_c),]))
        for m in self.modules():
            self.weight_init(m)
        self.pool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)
        self.layer_name_num={}
        self.pruned_filters={}
        self.remaining_filters={}
        self.remaining_filters_each_epoch=[]

    def forward(self, x):
        # Encoder 1
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        pool1 = self.pool(layer2)
        # Encoder 2
        layer3 = self.layer3(pool1)
        layer4 = self.layer4(layer3)
        pool2 = self.pool(layer4)
        # Encoder 3
        layer5 = self.layer5(pool2)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        pool3 = self.pool(layer7)
        # Encoder 4
        layer8 = self.layer8(pool3)
        layer9 = self.layer9(layer8)
        layer10 = self.layer10(layer9)
        pool4 = self.pool(layer10)
        # Encoder 5
        layer11 = self.layer11(pool4)
        layer12 = self.layer12(layer11)
        layer13 = self.layer13(layer12)
        #pool5 = self.pool(layer13)
        #avgpool 
        avg_x=nn.AvgPool2d(2)(layer13)
        # Classifier
        fc1 = self.fc1(avg_x.view(avg_x.size(0), -1))
        classifier = self.classifier(fc1)
        return classifier

# Model, optimizer, and scheduler
model = VGG16(n_c=10, a_type='relu').to(device)
optimizer = th.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#optimizer = th.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 140, 230], gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Load pre-trained model if available
checkpoint = th.load('initial_pruning.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])
epoch_train_acc = checkpoint['train_acc']
epoch_test_acc = checkpoint['test_acc']

def calculate_cosine_similarity(layer_weights):
    num_filters = layer_weights.size(0)
    flat_filters = layer_weights.view(num_filters, -1)
    cosine_sim = CosineSimilarity(dim=1)
    sim_matrix = th.zeros(num_filters, num_filters, device=layer_weights.device)
    for i in range(num_filters):
        sim_values = cosine_sim(flat_filters[i].unsqueeze(0), flat_filters)
        sim_matrix[i] = sim_values
    return sim_matrix

def calculate_decorrelation_term(sim_matrix, lower_bound=0.3, upper_bound=0.4):
    weak_corr = ((sim_matrix > lower_bound) & (sim_matrix < upper_bound)).float()
    decorrelation_term = weak_corr.sum().item()
    return decorrelation_term

# Get convolutional layers
conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]

# Pruning loop
continue_pruning = True
prunes = 0
best_train_acc = epoch_train_acc
best_test_acc = epoch_test_acc

# Create CSV file and write the header
with open(csv_file_path, mode='w', newline='') as file:
    writer_csv = csv.writer(file)
    writer_csv.writerow(['Iteration', 'Epoch', 'Loss with Regularization', 'CrossEntropy Loss', 'Regularization Term', 'Decorrelation Term','Accuracy', 
                 'Conv 1', 'Conv 2', 'Conv 3', 'Conv 4', 'Conv 5', 'Conv 6', 'Conv 7', 'Conv 8', 'Conv 9', 'Conv 10', 'Conv 11', 'Conv 12', 'Conv 13',
                 'Trainable Parameters', 'Total Flops'])

while continue_pruning:
    # Load the best model from the previous iteration
    if prunes > 0:
        checkpoint = th.load('best_model_new.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch_train_acc = checkpoint['train_acc']
        epoch_test_acc = checkpoint['test_acc']

    layer_similarities = []
    selected_indices = [[] for _ in range(len(conv_layers))]
    remaining_indices = [[] for _ in range(len(conv_layers))]
    
    for i, layer in enumerate(conv_layers):
        with th.no_grad():
            filters = layer.weight.data.clone()
            num_filters = filters.size(0)
            
            # Calculate cosine similarity
            cosine_sim = CosineSimilarity(dim=0)
            similarity_matrix = []
            for j in range(num_filters):
                for k in range(j + 1, num_filters):
                    sim = cosine_sim(filters[j].flatten().unsqueeze(0), filters[k].flatten().unsqueeze(0))
                    similarity_matrix.append((j, k, sim.item()))
            
            # Sort by cosine similarity (descending)
            similarity_matrix.sort(key=lambda x: -x[2])
            layer_similarities.append(similarity_matrix)
            
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
                    sim_matrix = calculate_cosine_similarity(filters)

                    # Sort similarity values and select the top 2%
                    sim_values = sim_matrix.view(-1)
                    top_2_percent_idx = th.topk(sim_values, int(regularization_prune_percentage * sim_values.numel()), largest=True).indices
                    regularization_term += th.exp(-th.sum(sim_values[top_2_percent_idx]))

                    # Calculate decorrelation term
                    decorrelation_value = calculate_decorrelation_term(sim_matrix, decorrelation_lower_bound, decorrelation_higher_bound)
                    decorrelation_term += th.exp(-th.tensor(decorrelation_value, device=device))

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

        test_acc = []
        
        with th.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                y_hat = th.argmax(output, 1)
                test_acc.append((y_hat == targets).sum().item())
        epoch_test_acc = sum(test_acc) * 100 / len(testloader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Test Accuracy: {epoch_test_acc:.2f}%')

        if epoch_test_acc > best_test_acc:
            best_train_acc = epoch_train_acc
            best_test_acc = epoch_test_acc
            best_model_wts = model.state_dict()
            best_opt_wts = optimizer.state_dict()
            best_sch_wts = scheduler.state_dict()
            csv_data = [prunes, epoch + 1, (running_loss / len(trainloader)), (old_running_loss / len(trainloader)), regularization_term.item(), 
                        decorrelation_term.item(), best_test_acc]
    
    # Print remaining filters in each convolutional layer
    for i, layer in enumerate(conv_layers):
        print(f'Layer {i+1} - Remaining Filters: {layer.out_channels}')
        csv_data.append(layer.out_channels)
    
    print('csv_data:',csv_data)
    # Write data to CSV
    with open(csv_file_path, mode='a', newline='') as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(csv_data)
    
    # Prune filters
    model.prune_filters(remaining_indices)
    
    # Fine tuning
    for epoch in range(epochs):
        train_acc = []
        best_train_acc, best_test_acc = 0, 0
        old_running_loss, running_loss = 0, 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Calculate finetuning term
            finetuning_term = 0

            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    filters = module.weight
                    l2_norm_value = th.norm(filters, p=2).item()
                    finetuning_term += l2_norm_value
            print(f"Sum of L2 norms: {l2_norm_sum}")

            log_term = math.log(l2_norm_sum + 1e-8)  # Adding a small epsilon to avoid log(0)
            print(f"Log of sum: {log_term}")

            finetuning_term = math.exp(-log_term)
            print(f"Finetuning term (e^-log_term): {finetuning_term}")

            finetuning_term = th.tensor(finetuning_term, device=device)  # Convert to tensor
            print(f"Finetuning term as tensor: {finetuning_term}")

            output = model(inputs)
            old_loss = criterion(output, targets)
            old_running_loss += old_loss.item()
            
            new_loss = old_loss + (delta * finetuning_term)
            new_loss.backward()
            optimizer.step()
            running_loss += new_loss.item()

            with th.no_grad():
                y_hat = th.argmax(output, 1)
                train_acc.append((y_hat == targets).sum().item())

        epoch_train_acc = sum(train_acc) * 100 / len(trainloader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Train Accuracy: {epoch_train_acc:.2f}%')
        print(f'Finetuning Iteration {prunes + 1}, Epoch [{epoch + 1}/{epochs}], Old Loss: {old_running_loss / len(trainloader):.8f}, New Loss: {running_loss / len(trainloader):.8f}')
        print(f'Finetuning term: {finetuning_term}')

        test_acc = []
        
        with th.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                y_hat = th.argmax(output, 1)
                test_acc.append((y_hat == targets).sum().item())
        epoch_test_acc = sum(test_acc) * 100 / len(testloader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Test Accuracy: {epoch_test_acc:.2f}%')

        if epoch_test_acc > best_test_acc:
            best_train_acc = epoch_train_acc
            best_test_acc = epoch_test_acc
            best_model_wts = model.state_dict()
            best_opt_wts = optimizer.state_dict()
            best_sch_wts = scheduler.state_dict()
    
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
summary(model, (3, 32, 32))

# Close TensorBoard writer
#writer.close()