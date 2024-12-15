import torch
import numpy as np
import matplotlib.pyplot as plt
from triangle import triangulate
import math
from testfunction import TestFunction
import quadpy

class element:
    def __init__(self, vertex) -> None:
        self.face = np.zeros(3, dtype=int) # global number of faces
        self.reftype = -1 # -1 for inactive element and 0 for active element
        self.vertex = vertex # np.zeros(3, dtype=int)

class face:
    def __init__(self, vertex, neighbor) -> None:
        self.vertex = vertex # np.zeros(2, dtype=int) # global number of vertices
        self.neighbor = neighbor # np.zeros(2, dtype=int) # global number of vertices


class GenData:
    def __init__(self, name:str, param:str) -> None:
        # 初始化顶点、边和参数
        if name=='regular':
            self.v = [[0, 0], [0, 1], [1, 1], [1, 0]]
            self.segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
        elif name=='irregular':
            self.v = [[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]]
            self.segments = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
        else:
            return NotImplementedError
        self.para = param
        self.np_dtype = np.float64
        self.torch_dtype = torch.float32
        
        # 使用 triangulate 生成三角形网格
        t = triangulate({"vertices": self.v, 'segments': self.segments}, self.para)
        print('ok')
        self.Mesh = t["triangles"]
        # print(t['segments'])
        self.points = torch.tensor(t["vertices"], dtype=torch.float32)
        self.edges = t["edges"]
        # print(self.edges[0],self.Mesh[0])
        
        # 记录网格属性
        self.Nv = len(self.points)  # 顶点数量
        self.Nelt = len(self.Mesh)  # 元素数量
        self.Nedge = len(self.edges)  # 边数量
        self.Nif = 0  # 内部边数量
        self.Nbf = 0  # 边界边数量

        self.bdEdge = []
        self.inEdge = []

        # 遍历每条边，查找相邻的单元格 用于统计每条边的两边是哪两个单元，如果只有一个单元那代表是边界边
        for i, edge in enumerate(self.edges):
            neighbor = [k for k in range(self.Nelt) if edge[0] in self.Mesh[k] and edge[1] in self.Mesh[k]]
            # print(neighbor)
            
            if len(neighbor) == 2:  # 内部边
                e = face(edge, neighbor)
                self.inEdge.append(e)
                self.Nif += 1
            elif len(neighbor) == 1:  # 边界边
                e = face(edge, neighbor)
                self.bdEdge.append(e)
                self.Nbf += 1

        # 打印网格信息
        print(f'In the whole domain: ')
        print(f'{self.Nv} points')
        print(f'{self.Nelt} elements')
        print(f'{self.Nedge} faces/edges')
        print(f'{self.Nif} interior edges, {self.Nbf} boundary edges')


    
    def plot_mesh(self):
        '''
        plot the mesh
        '''
        print("Plot the mesh:")
        plt.triplot(self.points[:,0], self.points[:,1], self.Mesh) 
        plt.plot(self.points[:,0], self.points[:,1], 'o', markersize=1) 
        plt.axis('equal')
        plt.savefig('mesh.png') 
        plt.show()

    def test_function(self, mesh, deg):
        '''
        Calculate the values and derivatives of test functions
        
        Args:
            mesh: mesh object
            deg: degree of test functions
            
        Returns:
            v: values of test functions on each element, shape [Nelt, deg+1]
            dv: derivatives of test functions on each element, shape [Nelt, deg+1, 2]
        '''
        testfunc = TestFunction()
        v, dv = [], []
        for i in range(self.Nelt):
            elt = self.Mesh[i]
            p1 = self.points[elt[0]]; p2 = self.points[elt[1]]; p3 = self.points[elt[2]]; 
            BE, _ = self.computeBE(p1, p2, p3)
            invB = torch.inverse(BE)
            vi, dvi = [], []
            for k in range(deg + 1):
                vk, dvk = testfunc.get_value(mesh, k)
                dvk = torch.matmul(dvk, invB)
                # print(dvk.shape)
                vi.append(vk)
                dvi.append(dvk)
            v.append(vi)
            dv.append(dvi)
        return v, dv
            

    
    def Grids_elt(self, Nint_elt: int):
        '''
        给出每个单元上的积分点，步骤是先给出参考单元上的积分点，然后 通过仿射变换，之后变换到问题的网格单元上； 以及每个单元的面积
        output: 
            Grid: N_elt, N of int_elt, 2
            area: N_elt
        '''
        self.Grid = []
        self.Area = []
        p = self.get_grid_refelt(Nint_elt)
        for i in range(self.Nelt):
            elt = self.Mesh[i]
            p1 = self.points[elt[0]]; p2 = self.points[elt[1]]; p3 = self.points[elt[2]]; 
            BE, bE = self.computeBE(p1, p2, p3)
            area = torch.det(BE) / 2.
            P = torch.matmul(p, BE.T) + bE
            # print(p, BE, bE, P)
            self.Grid.append(P)
            self.Area.append(area)
        return torch.stack(self.Grid), torch.stack(self.Area)
    
    def Grids_edge(self, Nint_edge: int):
        '''
        由于需要做边界积分，给出从参考边上到实际边上的映射关系；最后给出的是每条边上的积分点坐标，每条边的法向量，每条边的长度
        return 
            Points:         Nelt * 3 * N of int_edge * 2, 
            Norm vector:    Nelt * 3 * 2
            length of edge: Nelt * 3
        '''
        self.Grid = []
        self.Nvec = []
        self.L = []
        for i in range(self.Nelt):
            elt = self.Mesh[i]
            p1 = self.points[elt[0]]; p2 = self.points[elt[1]]; p3 = self.points[elt[2]]; 
            grid, nvec, l = self.get_int_edge(p1, p2, p3, Nint_edge)
            self.Grid.append(grid)
            self.Nvec.append(nvec)
            self.L.append(l)
        return torch.stack(self.Grid), torch.stack(self.Nvec), torch.stack(self.L)      

    def Grids_bd_edge(self, N_bd):
        '''
        给出边界上的边上的积分点以及其唯一的相邻单元编号
        '''
        points = []
        neigh = []
        for i in range(self.Nbf):
            e = self.bdEdge[i]
            vertex = e.vertex
            n = e.neighbor[0]
            p1 = self.points[vertex[0]]; p2 = self.points[vertex[1]]; 
            t = torch.linspace(0, 1, N_bd)
            p = (1 - t).unsqueeze(-1) * p1 + t.unsqueeze(-1) * p2
            points.append(p)
            neigh.append(n)
        return torch.stack(points), neigh

    def Grids_inner_edge(self, N_test):
        '''
        给出区域内部边上的积分点以及其相邻单元编号，以及法向量
        '''
        points = []
        nvecs = []
        neigh = []
        for i in range(self.Nif):
            e = self.inEdge[i]
            vertex = e.vertex
            n = e.neighbor
            p1 = self.points[vertex[0]]; p2 = self.points[vertex[1]]; 
            vec = p2 - p1
            nvec = torch.tensor([vec[1], -vec[0]]); nvec = nvec / torch.norm(nvec)
            t = torch.linspace(0, 1, N_test)
            p = (1 - t).unsqueeze(-1) * p1 + t.unsqueeze(-1) * p2
            points.append(p)
            nvecs.append(nvec)
            neigh.append(n)
        return torch.stack(points), torch.stack(nvecs), neigh


    
    def computeBE(self, p1, p2, p3):
        '''
        用于计算参考单元到实际单元的仿射变换
        '''
        BE = torch.tensor([[p2[0]-p1[0], p3[0]-p1[0]], [p2[1]-p1[1], p3[1]-p1[1]]], dtype=torch.float32)
        bE = p1.clone()
        return BE, bE
    

    def get_grid_refelt(self, Nint_elt: int):
        '''g给出参考单元上的积分点'''
        x = np.linspace(0, 1, Nint_elt, dtype=self.np_dtype)
        y = np.linspace(0, 1, Nint_elt, dtype=self.np_dtype)
        X, Y = np.meshgrid(x, y)
        mask = 1 - X - Y > -1e-9
        points = np.column_stack((X[mask], Y[mask]))
        return torch.tensor(points, dtype=self.torch_dtype).view(-1, 2)
    
    

    
    def get_grid_refedge(self, Nint_edge):
        '''给出参考单元上边长上的积分点'''
        t = torch.linspace(0, 1, Nint_edge)
        edge1 = torch.stack([t, torch.zeros_like(t)], dim=1) 
        edge2 = torch.stack([1 - t, t], dim=1) 
        edge3 = torch.stack([torch.zeros_like(t), 1 - t], dim=1)
        edges = torch.stack([edge1, edge2, edge3], dim=0)
        return edges
    
    # 注：这里面的积分点其实都可以换成Gaussian积分点，但在这里为了简单，直接使用等分点, 
    # 在实际使用中，可以换成Gaussian积分点，这样精度会更高,但我没试过，你可以试一下，直接掉包就可以应该


    
    def get_int_edge(self, p1:torch.tensor, p2:torch.tensor, p3:torch.tensor, Nint_edge: int):
        '''
        给出每个单元上边长上的积分点，以及其法向量，以及每条边的长度
        '''
        t = torch.tensor(np.linspace(0, 1, Nint_edge), dtype=self.torch_dtype).view(-1, 1)
        vec1 = p2 - p1; vec2 = p3 - p2; vec3 = p1 - p3
        norm1 = torch.tensor([vec1[1], -vec1[0]]); norm1 = norm1 / torch.norm(norm1)
        norm2 = torch.tensor([vec2[1], -vec2[0]]); norm2 = norm2 / torch.norm(norm2)
        norm3 = torch.tensor([vec3[1], -vec3[0]]); norm3 = norm3 / torch.norm(norm3)
        l1 = torch.norm(vec1); l2 = torch.norm(vec2); l3 = torch.norm(vec3)
        points1 = t * vec1 + p1; points2 = t * vec2 + p2; points3 = t * vec3 + p3
        points = torch.stack([points1, points2, points3]) 
        norm_vecs = torch.stack([norm1, norm2, norm3])
        l = torch.stack([l1, l2, l3])
        return points, norm_vecs, l




if __name__ == '__main__':
    gd = GenData(name='regular', param='pq30a0.1e')
    gd.plot_mesh()

    # p = gd.get_grid_refelt(5)
    # v, dv = gd.test_function(p, 2)
    # gd.plot_mesh()
    # def f(x):
    #     return np.sin(x[0]) * np.sin(x[1])


    # triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.7, 0.5]])

    # # get a "good" scheme of degree 10
    # scheme = quadpy.t2.get_good_scheme(10)
    # val = scheme.integrate(f, triangle)

    