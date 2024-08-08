import random
import os
import torch as th
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

# Set random seed for reproducibility
seed = 1787
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True

device = 'cuda' if th.cuda.is_available() else 'cpu'

batch_size_tr = 100
batch_size_te = 100

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

activation = 'relu'
n_classes = 10
epochs = 300

# Model, criterion, optimizer, and scheduler
model = VGG16(n_c=10, a_type='relu').to(device)
criterion = nn.CrossEntropyLoss()
optimizer = th.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[80, 140, 230], gamma=0.1)

best_train_acc = 0
best_test_acc = 0

for epoch in range(epochs):
    model.train()
    train_acc = []
    for batch_num, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda() if th.cuda.is_available() else (inputs, targets)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        with th.no_grad():
            _, predicted = th.max(outputs.data, 1)
            train_acc.append((predicted == targets).sum().item())

    epoch_train_acc = 100.0 * sum(train_acc) / len(trainset)
    if epoch_train_acc > best_train_acc:
        best_train_acc = epoch_train_acc

    model.eval()
    test_acc = []
    with th.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.cuda(), targets.cuda() if th.cuda.is_available() else (inputs, targets)
            outputs = model(inputs)
            _, predicted = th.max(outputs.data, 1)
            test_acc.append((predicted == targets).sum().item())

    epoch_test_acc = 100.0 * sum(test_acc) / len(testset)
    if epoch_test_acc > best_test_acc:
        best_test_acc = epoch_test_acc
        state = {
            'model': model.state_dict(),
            'train_acc': best_train_acc,
            'test_acc': best_test_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        th.save(state, 'vgg16_baseline_model.pth')

    scheduler.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Train Accuracy: {epoch_train_acc:.2f}%, Test Accuracy: {epoch_test_acc:.2f}%')

print(f'Best Train Accuracy: {best_train_acc:.2f}%')
print(f'Best Test Accuracy: {best_test_acc:.2f}%')
