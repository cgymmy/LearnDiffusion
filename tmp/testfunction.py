import torch

class TestFunction:
    def __init__(self, func_type='Polynomial'):
        self.type = func_type
    def get_value(self, mesh: torch.tensor, order: int=0):
        if self.type == 'Polynomial':
            return self.poly(mesh, order)
        else:
            raise NotImplementedError
    # def poly(self, mesh: torch.tensor, order: int):
    #     '''
    #     Input: 
    #         mesh: [N,2]
    #         order: order of polynomial
    #     Output:
    #         v: [order + 1, N]
    #         dv: [order+1, 2, N]
    #     '''
    #     x = mesh[:, 0]; y = mesh[:, 1]
    #     if order == 0:
    #         v = torch.stack([1+0*x], dim=0)
    #         dv = torch.stack([0*x, 0*y], dim=1)
    #         dv = torch.stack([dv], dim=0)
    #     elif order == 1:
    #         v = torch.stack([x, y], dim=0)
    #         dv_x = torch.stack([1+0*x, 0*y], dim=1)
    #         dv_y = torch.stack([0*x, 1+0*y], dim=1)
    #         dv = torch.stack([dv_x, dv_y], dim=0)  
    #     elif order == 2:
    #         v = torch.stack([x**2, y**2, x*y], dim=0)  
    #         dv1 = torch.stack([2*x, 0*y], dim=1)
    #         dv2 = torch.stack([0*x, 2*y], dim=1)
    #         dv3 = torch.stack([y, x], dim=1)
    #         dv = torch.stack([dv1, dv2, dv3], dim=0)
    #     else:
    #         return NotImplementedError
    #     return v, dv
    def poly(self, mesh: torch.tensor, order: int):
        '''
        Input: 
            mesh: [N,2]  # N是网格点的数量
            order: order of polynomial  # 多项式的阶数
        Output:
            v: [M, N]     # M是所有可能的 x^i y^j 的项数目，N是网格点数量
            dv: [M, 2, N] # M是所有多项式项数，2 表示对 x 和 y 的偏导数，N 是网格点数
        '''
        x = mesh[:, 0]
        y = mesh[:, 1]
        
        # 用来存储多项式和导数
        v_list = []
        dv_list = []
        
        for i in range(order + 1):
            for j in range(order + 1 - i):
                v_list.append(x**i * y**j)
                
                dv_x = i * x**(i-1) * y**j if i > 0 else torch.zeros_like(x)
                dv_y = j * x**i * y**(j-1) if j > 0 else torch.zeros_like(y)
                
                dv_list.append(torch.stack([dv_x, dv_y], dim=1))
        
        v = torch.stack(v_list, dim=0)   # [M, N] 多项式项数 M, 网格点数量 N
        dv = torch.stack(dv_list, dim=0) # [M, 2, N] M 是项数，2 是 (x, y) 两个方向的偏导数
        
        return v, dv


if __name__=='__main__':
    tf = TestFunction()
    mesh = torch.randn(5,2)
    v,dv = tf.get_value(mesh, 1)
    print(v.shape, dv.shape)