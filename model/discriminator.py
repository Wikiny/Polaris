from torch import nn   
class Discriminator(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.discriminator(x)