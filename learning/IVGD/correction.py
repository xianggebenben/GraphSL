import torch
import torch.nn.functional as F
class correction(torch.nn.Module):
    def __init__(self):
        super(correction, self).__init__()
        number_of_neurons =1000
        self.fc1 = torch.nn.Linear (2,number_of_neurons)
        self.fc2 = torch.nn.Linear(number_of_neurons, number_of_neurons)
        self.fc3= torch.nn.Linear(number_of_neurons,2)
    def forward(self, x):
        x= torch.cat((1-x,x),dim=1)
        temp = F.relu(self.fc1(x))
        temp = F.relu(self.fc2(temp))
        temp = self.fc3(temp)
        temp = (temp+x)
        temp = torch.minimum(torch.maximum(torch.zeros(temp.shape),temp),torch.ones(temp.shape))
        return temp