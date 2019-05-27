from torch import nn

#64x64
class MyAutoencoder64(nn.Module):
    def __init__(self, n_comp=10):
        super(MyAutoencoder64, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(4096, 500), nn.Sigmoid(), 
        nn.Linear(500, 64), nn.Sigmoid(), 
        nn.Linear(64, n_comp))
        self.decoder = nn.Sequential(nn.Linear(n_comp, 64), nn.Tanh(), 
        nn.Linear(64, 500), nn.Tanh(), 
        nn.Linear(500, 4096))
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)

        return x.data.numpy()

class MySparceAutoencoder64(nn.Module):
    def __init__(self, n_comp=10):
        super(MySparceAutoencoder64, self).__init__()
        self.n_comp = n_comp

        self.encoder = nn.Sequential(nn.Linear(4096, 1000), nn.Sigmoid(), 
        nn.Linear(1000, 500), nn.Sigmoid(), 
        nn.Linear(500, 100))
        self.decoder = nn.Sequential(nn.Linear(100, 500), nn.Tanh(), 
        nn.Linear(500, 1000), nn.Tanh(), 
        nn.Linear(1000, 4096))

    def forward(self, x):
        x = self.encoder(x)
        x[x < x.sort(dim=1, descending=True)[0][:,self.n_comp - 1].resize(len(x),1)] = 0
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x[x < x.sort(dim=1, descending=True)[0][:,self.n_comp - 1].resize(len(x),1)] = 0

        return x.data.numpy()

class MyDeepAutoencoder64(nn.Module):
    def __init__(self, n_comp=10):
        super(MyDeepAutoencoder64, self).__init__()
        self.n_comp = n_comp
        
        self.encoder = nn.Sequential(nn.Linear(4096, 2000), nn.Sigmoid(), 
        nn.Linear(2000, 1100), nn.Sigmoid(), 
        nn.Linear(1100, 500), nn.Sigmoid(), 
        nn.Linear(500, 300), nn.Sigmoid(), 
        nn.Linear(300, 100), nn.Sigmoid(), 
        nn.Linear(100, n_comp))
        self.decoder = nn.Sequential(nn.Linear(n_comp, 100), nn.Tanh(), 
        nn.Linear(100, 300), nn.Tanh(),
        nn.Linear(300, 500), nn.Tanh(),
        nn.Linear(500, 1100), nn.Tanh(),
        nn.Linear(1100, 2000), nn.Tanh(), 
        nn.Linear(2000, 4096))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)

        return x.data.numpy()
class MyDeepSparceAutoencoder64(nn.Module):
    def __init__(self, n_comp=10):
        super(MyDeepSparceAutoencoder64, self).__init__()
        self.n_comp = n_comp

        self.encoder = nn.Sequential(nn.Linear(4096, 2000), nn.Sigmoid(), 
        nn.Linear(2000, 1100), nn.Sigmoid(), 
        nn.Linear(1100, 500), nn.Sigmoid(), 
        nn.Linear(500, 300), nn.Sigmoid(), 
        nn.Linear(300, 100))
        self.decoder = nn.Sequential( 
        nn.Linear(100, 300), nn.Tanh(),
        nn.Linear(300, 500), nn.Tanh(),
        nn.Linear(500, 1100), nn.Tanh(),
        nn.Linear(1100, 2000), nn.Tanh(), 
        nn.Linear(2000, 4096))

    def forward(self, x):
        x = self.encoder(x)
        x[x < x.sort(dim=1, descending=True)[0][:,self.n_comp - 1].resize(len(x),1)] = 0
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x[x < x.sort(dim=1, descending=True)[0][:,self.n_comp - 1].resize(len(x),1)] = 0

        return x.data.numpy()
#128x128
class MyAutoencoder128(nn.Module):
    def __init__(self, n_comp=10):
        super(MyAutoencoder128, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(16384, 5000), nn.Sigmoid(), 
        nn.Linear(5000, 1000),nn.Sigmoid(), 
        nn.Linear(1000, 300), nn.Sigmoid(), 
        nn.Linear(300, 60), nn.Sigmoid(), 
        nn.Linear(60, n_comp))
        self.decoder = nn.Sequential(nn.Linear(n_comp, 60), nn.Tanh(), 
        nn.Linear(60, 300), nn.Tanh(),
        nn.Linear(300, 1000), nn.Tanh(),
        nn.Linear(1000, 5000), nn.Tanh(), 
        nn.Linear(5000, 16384))
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        
        return x.data.numpy()

class MySparceAutoencoder128(nn.Module):
    def __init__(self, n_comp=10):
        super(MySparceAutoencoder128, self).__init__()
        self.n_comp = n_comp
        self.encoder = nn.Sequential(nn.Linear(16384, 5000), nn.Sigmoid(), 
        nn.Linear(5000, 1000),nn.Sigmoid(), 
        nn.Linear(1000, 300), nn.Sigmoid(), 
        nn.Linear(300, 100))
        self.decoder = nn.Sequential(nn.Linear(100, 300), nn.Tanh(), 
        nn.Linear(300, 1000), nn.Tanh(),
        nn.Linear(1000, 5000), nn.Tanh(), 
        nn.Linear(5000, 16384))

    def forward(self, x):
        x = self.encoder(x)
        x[x < x.sort(dim=1, descending=True)[0][:,self.n_comp - 1].resize(len(x),1)] = 0
        x = self.decoder(x)

        return x

    def encode(self, x):
        x = self.encoder(x)
        x[x < x.sort(dim=1, descending=True)[0][:,self.n_comp - 1].resize(len(x),1)] = 0

        return x.data.numpy()

class MyDeepAutoencoder128(nn.Module):
    def __init__(self, n_comp=10):
        super(MyDeepAutoencoder128, self).__init__()
        self.n_comp = n_comp
        self.encoder = nn.Sequential(nn.Linear(16384, 10000), nn.Sigmoid(), 
        nn.Linear(10000, 6000),nn.Sigmoid(), 
        nn.Linear(6000, 2000), nn.Sigmoid(), 
        nn.Linear(2000, 800), nn.Sigmoid(), 
        nn.Linear(800, 300), nn.Sigmoid(), 
        nn.Linear(300, 100), nn.Sigmoid(), 
        nn.Linear(100, n_comp))
        self.decoder = nn.Sequential(nn.Linear(n_comp, 100), nn.Tanh(), 
        nn.Linear(100, 300), nn.Tanh(),
        nn.Linear(300, 800), nn.Tanh(),
        nn.Linear(800, 2000), nn.Tanh(),
        nn.Linear(2000, 6000), nn.Tanh(),
        nn.Linear(6000, 10000), nn.Tanh(), 
        nn.Linear(10000, 16384))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def encode(self, x):
        x = self.encoder(x)
        return x.data.numpy()

class MyDeepSparceAutoencoder128(nn.Module):
    def __init__(self, n_comp=10):
        super(MyDeepSparceAutoencoder128, self).__init__()
        self.n_comp = n_comp
        self.encoder = nn.Sequential(nn.Linear(16384, 10000), nn.Sigmoid(), 
        nn.Linear(10000, 6000),nn.Sigmoid(), 
        nn.Linear(6000, 2000), nn.Sigmoid(), 
        nn.Linear(2000, 800), nn.Sigmoid(), 
        nn.Linear(800, 300), nn.Sigmoid(), 
        nn.Linear(300, 100))
        self.decoder = nn.Sequential(nn.Linear(100, 300), nn.Tanh(),
        nn.Linear(300, 800), nn.Tanh(),
        nn.Linear(800, 2000), nn.Tanh(),
        nn.Linear(2000, 6000), nn.Tanh(),
        nn.Linear(6000, 10000), nn.Tanh(), 
        nn.Linear(10000, 16384))

    def forward(self, x):
        x = self.encoder(x)
        x[x < x.sort(dim=1, descending=True)[0][:,self.n_comp - 1].resize(len(x),1)] = 0
        x = self.decoder(x)
        
        return x

    def encode(self, x):
        x = self.encoder(x)
        x[x < x.sort(dim=1, descending=True)[0][:,self.n_comp - 1].resize(len(x),1)] = 0

        return x


class MyCAutoencoder(nn.Module):
    def __init__(self, n_comp=16):
        super(MyCAutoencoder, self).__init__()
        self.n_comp = n_comp
        self.ec1 = nn.Conv2d(1,2,(3,3),padding=1) 
        self.ep1 = nn.MaxPool2d(4, return_indices=True) 
        self.ec2 = nn.Conv2d(2,4,(3,3),padding=1)
        self.ep2 = nn.MaxPool2d(2, return_indices=True)
        self.ec3 = nn.Conv2d(4,n_comp,(3,3),padding=1)
        self.ep3 = nn.MaxPool2d(4, return_indices=True)
        self.el = nn.Linear(4,4)
        self.ep4 = nn.MaxPool2d(4, return_indices=True)
        
        self.dp4 = nn.MaxUnpool2d(4)
        self.dl = nn.Linear(4,4)
        self.dp3 = nn.MaxUnpool2d(4)
        self.dc3 = nn.Conv2d(n_comp,4,(3,3),padding=1) 
        self.dp2 = nn.MaxUnpool2d(2)
        self.dc2 = nn.Conv2d(4,2,(3,3),padding=1)
        self.dp1 = nn.MaxUnpool2d(4)
        self.dc1 = nn.Conv2d(2,1,(3,3),padding=1)
        
    def forward(self, x):
        relu = nn.ReLU()
        #encode
        x = self.ec1(x)
        x, i1 = self.ep1(x)
        x = relu(x)
        x = self.ec2(x)
        x, i2 = self.ep2(x)
        x = relu(x)
        x = self.ec3(x)
        x, i3 = self.ep3(x)
        x = relu(x)
        x = self.el(x)
        x, i4 = self.ep4(x)
        
        x = self.dp4(x, i4)
        x = self.dl(x)
        x = relu(x)
        x = self.dp3(x, i3)
        x = self.dc3(x)
        x = relu(x)
        x = self.dp2(x, i2)
        x = self.dc2(x)
        x = relu(x)
        x = self.dp1(x, i1)
        x = self.dc1(x)

        
        return x
    def encode(self, x):
        relu = nn.ReLU()
        
        x = self.ec1(x)
        x, i1 = self.ep1(x)
        x = relu(x)
        x = self.ec2(x)
        x, i2 = self.ep2(x)
        x = relu(x)
        x = self.ec3(x)
        x, i3 = self.ep3(x)
        x = relu(x)
        x = self.el(x)
        x, i4 = self.ep4(x)
        
        return x.data.numpy().reshape(-1, self.n_comp)

