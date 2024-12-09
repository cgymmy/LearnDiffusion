import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
class ChannelShuffle(nn.Module):
    def __init__(self,groups):
        super().__init__()
        self.groups=groups
    def forward(self,x):
        n,c,h,w=x.shape
        x=x.view(n,self.groups,c//self.groups,h,w) # group
        x=x.transpose(1,2).contiguous().view(n,-1,h,w) #shuffle
        return x

class ConvBnSiLu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0):
        super().__init__()
        self.module=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.SiLU(inplace=True))
    def forward(self,x):
        return self.module(x)

class ResidualBottleneck(nn.Module):
    '''
    shufflenet_v2 basic unit(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.branch1=nn.Sequential(nn.Conv2d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
                                    nn.BatchNorm2d(in_channels//2),
                                    ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))
        self.branch2=nn.Sequential(ConvBnSiLu(in_channels//2,in_channels//2,1,1,0),
                                    nn.Conv2d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
                                    nn.BatchNorm2d(in_channels//2),
                                    ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))
        self.channel_shuffle=ChannelShuffle(2)

    def forward(self,x):
        x1,x2=x.chunk(2,dim=1)
        x=torch.cat([self.branch1(x1),self.branch2(x2)],dim=1)
        x=self.channel_shuffle(x) #shuffle two branches

        return x

class ResidualDownsample(nn.Module):
    '''
    shufflenet_v2 unit for spatial down sampling(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.branch1=nn.Sequential(nn.Conv2d(in_channels,in_channels,3,2,1,groups=in_channels),
                                    nn.BatchNorm2d(in_channels),
                                    ConvBnSiLu(in_channels,out_channels//2,1,1,0))
        self.branch2=nn.Sequential(ConvBnSiLu(in_channels,out_channels//2,1,1,0),
                                    nn.Conv2d(out_channels//2,out_channels//2,3,2,1,groups=out_channels//2),
                                    nn.BatchNorm2d(out_channels//2),
                                    ConvBnSiLu(out_channels//2,out_channels//2,1,1,0))
        self.channel_shuffle=ChannelShuffle(2)

    def forward(self,x):
        x=torch.cat([self.branch1(x),self.branch2(x)],dim=1)
        x=self.channel_shuffle(x) #shuffle two branches

        return x

class TimeMLP(nn.Module):
    '''
    naive introduce timestep information to feature maps with mlp and add shortcut
    '''
    def __init__(self,embedding_dim,hidden_dim,out_dim):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(embedding_dim,hidden_dim),
                                nn.SiLU(),
                               nn.Linear(hidden_dim,out_dim))
        self.act=nn.SiLU()
    def forward(self,x,t):
        t_emb=self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        x=x+t_emb
  
        return self.act(x)
    
class EncoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels,out_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=out_channels,out_dim=out_channels//2)
        self.conv1=ResidualDownsample(out_channels//2,out_channels)
    
    def forward(self,x,t=None):
        x_shortcut=self.conv0(x)
        if t is not None:
            x=self.time_mlp(x_shortcut,t)
        x=self.conv1(x)

        return [x,x_shortcut]
        
class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels,in_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=in_channels,out_dim=in_channels//2)
        self.conv1=ResidualBottleneck(in_channels//2,out_channels//2)

    def forward(self,x,x_shortcut,t=None):
        x=self.upsample(x)
        x=torch.cat([x,x_shortcut],dim=1)
        x=self.conv0(x)
        if t is not None:
            x=self.time_mlp(x,t)
        x=self.conv1(x)

        return x        

class Unet(nn.Module):
    '''
    simple unet design without attention
    '''
    def __init__(self,timesteps,time_embedding_dim,in_channels=3,out_channels=2,base_dim=32,dim_mults=[2,4,8,16], temp: float = 20.0):
        super().__init__()
        assert isinstance(dim_mults,(list,tuple))
        assert base_dim%2==0 

        channels=self._cal_channels(base_dim,dim_mults)

        self.init_conv=ConvBnSiLu(in_channels,base_dim,3,1,1)
        #self.time_embedding=nn.Embedding(timesteps,time_embedding_dim)
        self.time_embedding=nn.Linear(timesteps,time_embedding_dim)

        self.encoder_blocks=nn.ModuleList([EncoderBlock(c[0],c[1],time_embedding_dim) for c in channels])
        self.decoder_blocks=nn.ModuleList([DecoderBlock(c[1],c[0],time_embedding_dim) for c in channels[::-1]])
    
        self.mid_block=nn.Sequential(*[ResidualBottleneck(channels[-1][1],channels[-1][1]) for i in range(2)],
                                        ResidualBottleneck(channels[-1][1],channels[-1][1]//2))

        self.final_conv=nn.Conv2d(in_channels=channels[0][0]//2,out_channels=out_channels,kernel_size=1)

        self.centers = nn.Parameter(torch.linspace(0,1,timesteps+1)[:-1]+0.5/timesteps,requires_grad=False)
        self.temp = temp
        

    def get_softmax(self, t):
        softmax_mat = F.softmax(-self.temp*torch.abs(t[:,None]-self.centers[None,:]), dim=1)
        return softmax_mat
        
    def get_time_emb(self,t):
        softmax_mat = self.get_softmax(t)
        t=self.time_embedding(softmax_mat)
    
        return t
        
    def forward(self,x,t=None):
        
        x=self.init_conv(x)
        
        if t is not None:
            t = self.get_time_emb(t)
            
        encoder_shortcuts=[]
        for encoder_block in self.encoder_blocks:
            x,x_shortcut=encoder_block(x,t)
            encoder_shortcuts.append(x_shortcut)
        x=self.mid_block(x)
        encoder_shortcuts.reverse()
        for decoder_block,shortcut in zip(self.decoder_blocks,encoder_shortcuts):
            x=decoder_block(x,shortcut,t)
        x=self.final_conv(x)

        return x

    def _cal_channels(self,base_dim,dim_mults):
        dims=[base_dim*x for x in dim_mults]
        dims.insert(0,base_dim)
        channels=[]
        for i in range(len(dims)-1):
            channels.append((dims[i],dims[i+1])) # in_channel, out_channel

        return channels

################################################################################################
class ItoSDE:
    def __init__(self, T_max: float):
        self.T_max = T_max
    def cond_exp(self, X_0: torch.Tensor, t: torch.Tensor):
        return NotImplementedError
    def cond_var(self, X_0: torch.Tensor, t: torch.Tensor):
        return NotImplementedError
    def f_drift(self, X_t: torch.Tensor, t: torch.Tensor):
        return NotImplementedError
    def g_random(self, X_t: torch.Tensor, t: torch.Tensor):
        return NotImplementedError
    def cond_std(self, X_0: torch.Tensor, t: torch.Tensor):
        return torch.sqrt(self.cond_var(X_0=X_0, t=t))
    
    def sample_random_times(self, length: int):
        return torch.rand(length) * self.T_max
    def run_forward(self, X_0: torch.Tensor, t: torch.Tensor):
        noise = torch.randn(X_0.shape)
        cond_std = self.cond_std(X_0, t)
        cond_exp = self.cond_exp(X_0, t)
        X_t = cond_exp + mult_first_dim(noise, cond_std)
        cond_std = torch.clip(cond_std, min=0.01)
        score = -mult_first_dim(noise, 1 / cond_std)
        return X_t, noise, score
    def run_forward_random_time(self, X_0: torch.Tensor):
        t = self.sample_random_times(X_0.shape[0])
        X_t, noise, score = self.run_forward(X_0=X_0, t=t)
        return X_t, noise, score, t

def mult_first_dim(X: torch.Tensor, t: torch.Tensor):
    return t.view(-1,*[1]*(X.dim()-1))*X

class VPSDE(ItoSDE):
    def __init__(self, T_max: float, beta_min: float=0.0, beta_max: float=1.0):
        super().__init__(T_max)
        self.beta_min = beta_min
        self.beta_max = beta_max
    def _beta_prime(self, t: torch.Tensor):
        return self.beta_min + (self.beta_max - self.beta_min)*t
    def _beta(self, t: torch.Tensor):
        return (self.beta_min*t) + 0.5*(self.beta_max - self.beta_min)*(t**2)
    def cond_exp(self, X_0, t):
        beta_t = self._beta(t)
        cond_exp_t = torch.exp(-0.5 * beta_t)
        return mult_first_dim(X_0, cond_exp_t)
    def cond_var(self, X_0, t):
        beta_t = self._beta(t)
        return 1- torch.exp(-beta_t)
    
    def f_drift(self, X_t, t):
        beta_prime = self._beta_prime(t)
        return -0.5 * mult_first_dim(X_t, beta_prime)
    def g_random(self, t):
        beta_prime = self._beta_prime(t)
        return torch.sqrt(beta_prime)

def run_backwards(model: torch.nn.Module, sde: VPSDE, X_0: torch.Tensor, device, train_score: bool=False, n_steps: int=10, plot_evolution: bool=True):
    model = model.to(device)
    n_traj = X_0.shape[0]
    time_grid = torch.linspace(sde.T_max, 0, n_steps)
    dt = torch.abs(time_grid[0] - time_grid[1])
    diffusion_g = sde.g_random(t=time_grid)
    noise = torch.randn(size=(n_steps, *list(X_0.shape)))
    diffusion_term_grid = mult_first_dim(noise, torch.sqrt(dt) * diffusion_g)
    x_traj = [X_0]
    if plot_evolution:
        n_col = len(time_grid)//2
        fig, axs = plt.subplots(2, n_col,figsize=(12*n_col, 24))
    for idx, t in enumerate(time_grid):
        x = x_traj[idx]
        t = t.repeat(n_traj)
        drift_term = sde.f_drift(x, t) * dt
        diffusion_term = diffusion_term_grid[idx]
        model_estimate = model(x.to(device), t.to(device)).detach().to('cpu')
        if train_score:
            score_estimates = model_estimate
        else:
            denominator = torch.clip(sde.cond_std(None, t), 0.01)
            score_estimates = -mult_first_dim(model_estimate, 1/denominator)
        g_squared = (diffusion_g[idx] ** 2).repeat(n_traj)
        
        correction_term = dt * mult_first_dim(score_estimates, g_squared)
        change = (correction_term - drift_term) + diffusion_term
        x_next = x + change
        x_traj.append(x_next)
        if plot_evolution:
            row = idx // n_col
            col = idx % n_col
            axs[row, col].scatter(x_next[:,0], x_next[:,1])
            axs[row, col].quiver(x_next[:,0], x_next[:,1], change[:,0],change[:,1])
            axs[row, col].set_xlim(-2.0,2.0)
            axs[row, col].set_ylim(-2.0,2.0)
            axs[row, col].set_title(f"Step={idx}")
    output = torch.stack(x_traj)
    return output, time_grid