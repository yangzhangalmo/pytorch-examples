''' This is a pytorch implementaion on CNN autoencoder
where the embedding layer is represented as a vector.
The code is mostly based on the implemenation of https://gist.github.com/okiriza/16ec1f29f5dd7b6d822a0a3f2af39274
'''

import random

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from   torchvision import datasets, transforms

class ae(nn.Module):
    
    def __init__(self, emb_size):
        
        super(ae, self).__init__()
        self.emb_size = emb_size

        # encoder components
        self.enc_cnn_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(4 * 4 * 20, 100)
        self.enc_linear_2 = nn.Linear(100, self.emb_size)

        # decoder components
        self.dec_linear_1 = nn.Linear(self.emb_size, 100)
        self.dec_linear_2 = nn.Linear(100, 20 * 20 * 20)
        self.dec_de_cnn_1 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.dec_de_cnn_2 = nn.ConvTranspose2d(10, 1, kernel_size=5)

    def encoder(self, images):
        ''' encoder construction
        '''

        emb = self.enc_cnn_1(images)
        emb = F.relu(F.max_pool2d(emb, 2))
        
        emb = self.enc_cnn_2(emb)
        emb = F.relu(F.max_pool2d(emb, 2))
        
        emb = emb.view([images.size(0), -1])

        emb = F.relu(self.enc_linear_1(emb))
        emb = F.relu(self.enc_linear_2(emb))

        return emb
    
    def decoder(self, emb):
        ''' decoder construction
        '''

        out = F.relu(self.dec_linear_1(emb))
        out = F.relu(self.dec_linear_2(out))

        out = out.view([emb.shape[0], 20, 20, 20])

        out = F.relu(self.dec_de_cnn_1(out))
        out = F.relu(self.dec_de_cnn_2(out))
        
        return out

    def forward(self, images):
        ''' auto encoder
        '''

        emb = self.encoder(images)
        out = self.decoder(emb)
        return out, emb

m = n = 28
emb_size = 100
num_epochs = 5
batch_size = 128
lr = 0.002

train_data = datasets.MNIST('~/data/mnist/', train=True , transform=transforms.ToTensor())
test_data  = datasets.MNIST('~/data/mnist/', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)

# Instantiate model
autoencoder = ae(emb_size)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    print("Epoch %d" % epoch)
    
    for i, (images, _) in enumerate(train_loader):    # Ignore image labels
        out, emb = autoencoder(Variable(images))
        
        optimizer.zero_grad()
        loss = loss_fn(out, images)
        loss.backward()
        optimizer.step()
        
    print("Loss = %.3f" % loss.item())

# Try reconstructing on test data
test_image = random.choice(test_data)[0]
test_image = Variable(test_image.view([1, 1, m, n]))
test_reconst, emb = autoencoder(test_image)

torchvision.utils.save_image(test_image.data, 'orig.png')
torchvision.utils.save_image(test_reconst.data, 'reconst.png')
