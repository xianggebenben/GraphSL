import torch
import torch.nn.functional as F

class correction(torch.nn.Module):
    """
    Define an error correction module.
    """

    def __init__(self):
        """
        Initializes the error correction module.
        """
        super(correction, self).__init__()
        number_of_neurons = 1000
        # Define the fully connected layers
        self.fc1 = torch.nn.Linear(2, number_of_neurons)
        self.fc2 = torch.nn.Linear(number_of_neurons, number_of_neurons)
        self.fc3 = torch.nn.Linear(number_of_neurons, 2)

    def forward(self, x):
        """
        Define the forward pass of the error correction module.

        Args:

        - x (torch.Tensor): Prediction of the seed vector from invertible graph residual net.

        Returns:

        - temp (torch.Tensor): Corrected prediction of the seed vector.
        """
        # Concatenate the input tensor with its complement along the second dimension
        x = torch.cat((1 - x, x), dim=1)
        # Apply the first fully connected layer followed by ReLU activation
        temp = F.relu(self.fc1(x))
        # Apply the second fully connected layer followed by ReLU activation
        temp = F.relu(self.fc2(temp))
        # Apply the third fully connected layer
        temp = self.fc3(temp)
        # Add the input tensor to the output tensor
        temp = (temp + x)
        # Clip the values of the output tensor between 0 and 1
        temp = torch.minimum(torch.maximum(torch.zeros(temp.shape).to(x.device), temp), torch.ones(temp.shape).to(x.device))
        return temp
