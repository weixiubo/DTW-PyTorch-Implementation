import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional, Tuple, Union

class DTW(nn.Module):
    """
    PyTorch实现的动态时间规整（Dynamic Time Warping）算子
    
    基于OpenDBA和cuDTW仓库实现，支持GPU加速和批处理
    """
    def __init__(self, 
                 use_cuda: bool = True,
                 normalize: bool = False,
                 sakoe_chiba_radius: Optional[int] = None):
        """
        初始化DTW模块
        
        参数:
            use_cuda: 是否使用CUDA加速计算（如果可用）
            normalize: 是否对DTW距离进行归一化（除以路径长度）
            sakoe_chiba_radius: Sakoe-Chiba带宽约束半径，用于加速计算，None表示不使用约束
        """
        super(DTW, self).__init__()
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.normalize = normalize
        self.sakoe_chiba_radius = sakoe_chiba_radius
    
    def _compute_dtw_matrix(self, 
                           x: torch.Tensor, 
                           y: torch.Tensor, 
                           dist_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算DTW矩阵
        
        参数:
            x: 第一个序列 [N, F]，其中N是序列长度，F是特征维度
            y: 第二个序列 [M, F]，其中M是序列长度，F是特征维度
            dist_mat: 可选的预计算距离矩阵 [N, M]
            
        返回:
            dtw_matrix: DTW累积距离矩阵 [N+1, M+1]
        """
        n, m = x.shape[0], y.shape[0]
        
        # 如果未提供距离矩阵，则计算欧氏距离矩阵
        if dist_mat is None:
            dist_mat = torch.cdist(x, y, p=2)  # [N, M]
        
        # 初始化DTW矩阵并填充无穷大值
        dtw_matrix = torch.full((n+1, m+1), float('inf'), device=x.device)
        dtw_matrix[0, 0] = 0.0
        
        # 初始化第一行和第一列（处理边界条件）
        for i in range(1, n+1):
            dtw_matrix[i, 0] = float('inf')  # 第一列保持无穷大
        
        for j in range(1, m+1):
            dtw_matrix[0, j] = float('inf')  # 第一行保持无穷大
        
        # 使用动态规划填充DTW矩阵
        for i in range(1, n+1):
            # 应用Sakoe-Chiba带宽约束（如果指定）
            j_start = max(1, i - self.sakoe_chiba_radius) if self.sakoe_chiba_radius else 1
            j_end = min(m+1, i + self.sakoe_chiba_radius + 1) if self.sakoe_chiba_radius else m+1
            
            for j in range(j_start, j_end):
                cost = dist_mat[i-1, j-1]
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # 垂直移动（插入）
                    dtw_matrix[i, j-1],    # 水平移动（删除）
                    dtw_matrix[i-1, j-1]   # 对角线移动（匹配）
                )
        
        return dtw_matrix
    
    def _compute_dtw_distance(self, 
                             x: torch.Tensor, 
                             y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算两个序列间的DTW距离和DTW矩阵
        
        参数:
            x: 第一个序列 [N, F]
            y: 第二个序列 [M, F]
            
        返回:
            distance: DTW距离
            dtw_matrix: DTW累积距离矩阵
        """
        dtw_matrix = self._compute_dtw_matrix(x, y)
        n, m = x.shape[0], y.shape[0]
        distance = dtw_matrix[n, m]
        
        # 如果需要归一化，除以路径长度
        if self.normalize:
            distance = distance / (n + m)
            
        return distance, dtw_matrix
    
    def _extract_optimal_path(self, dtw_matrix: torch.Tensor) -> torch.Tensor:
        """
        从DTW矩阵中提取最优对齐路径
        
        参数:
            dtw_matrix: DTW累积距离矩阵 [N+1, M+1]
            
        返回:
            path: 最优对齐路径，形状为 [P, 2]，其中P是路径长度
        """
        n, m = dtw_matrix.shape
        n -= 1
        m -= 1
        
        # 从右下角开始回溯
        path = [(n, m)]
        while n > 0 or m > 0:
            if n == 0:
                m -= 1
            elif m == 0:
                n -= 1
            else:
                min_val = min(dtw_matrix[n-1, m], 
                             dtw_matrix[n, m-1], 
                             dtw_matrix[n-1, m-1])
                
                if dtw_matrix[n-1, m-1] == min_val:
                    n -= 1
                    m -= 1
                elif dtw_matrix[n-1, m] == min_val:
                    n -= 1
                else:
                    m -= 1
            path.append((n, m))
        
        # 反转路径，使其从左上角开始
        path.reverse()
        return torch.tensor(path, device=dtw_matrix.device)
    
    def forward(self, 
               x: torch.Tensor, 
               y: Optional[torch.Tensor] = None, 
               return_path: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        计算DTW距离
        
        参数:
            x: 输入序列或序列批次
                - 单序列形状: [N, F]，其中N是序列长度，F是特征维度
                - 批次形状: [B, N, F]，其中B是批次大小
            y: 可选的第二个序列或序列批次，如果未提供，则计算x中每对序列间的距离
                - 单序列形状: [M, F]
                - 批次形状: [B, M, F]
            return_path: 是否返回最优对齐路径
            
        返回:
            distances: DTW距离
                - 单序列对: 标量张量
                - 批次: 形状为 [B] 的张量
            paths: 如果return_path=True，返回最优对齐路径
                - 单序列对: 形状为 [P, 2] 的张量
                - 批次: 形状为 [B] 的列表，每个元素是形状为 [P_i, 2] 的张量
        """
        # 确保输入在正确的设备上
        if self.use_cuda and x.device.type != 'cuda':
            x = x.cuda()
        
        # 处理批次输入
        if len(x.shape) == 3:  # [B, N, F]
            batch_size = x.shape[0]
            
            if y is None:
                # 如果未提供y，计算x中每个序列与自身的DTW距离（通常用于调试）
                y = x
            elif len(y.shape) == 2:  # [M, F]
                # 如果y是单个序列，将其扩展为批次
                y = y.unsqueeze(0).expand(batch_size, -1, -1)
            
            # 确保y在正确的设备上
            if self.use_cuda and y.device.type != 'cuda':
                y = y.cuda()
            
            # 批量计算DTW距离
            distances = torch.zeros(batch_size, device=x.device)
            paths = [] if return_path else None
            
            for i in range(batch_size):
                if return_path:
                    dist, dtw_mat = self._compute_dtw_distance(x[i], y[i])
                    distances[i] = dist
                    paths.append(self._extract_optimal_path(dtw_mat))
                else:
                    distances[i], _ = self._compute_dtw_distance(x[i], y[i])
            
            return (distances, paths) if return_path else distances
        
        # 处理单序列输入 [N, F]
        else:
            if y is None:
                y = x
            
            # 确保y在正确的设备上
            if self.use_cuda and y.device.type != 'cuda':
                y = y.cuda()
            
            dist, dtw_mat = self._compute_dtw_distance(x, y)
            
            if return_path:
                path = self._extract_optimal_path(dtw_mat)
                return dist, path
            else:
                return dist


class FastDTW(DTW):
    """
    改进的FastDTW实现，通过多分辨率方法加速DTW计算
    
    修复了低分辨率路径投影和窗口约束的问题
    """
    def __init__(self, 
                 radius: int = 1, 
                 use_cuda: bool = True,
                 normalize: bool = False):
        """
        初始化FastDTW模块
        
        参数:
            radius: FastDTW搜索半径
            use_cuda: 是否使用CUDA加速
            normalize: 是否归一化DTW距离
        """
        super(FastDTW, self).__init__(use_cuda=use_cuda, normalize=normalize)
        self.radius = radius
    
    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """
        简化的下采样实现，采用2倍下采样率
        """
        if x.shape[0] <= 2:
            return x
        
        # 简化的下采样方法 - 每隔一个点取一个，确保长度为偶数时正常工作
        return x[::2].clone()
    
    def _expand_window(self, path: torch.Tensor, radius: int, x_len: int, y_len: int) -> torch.Tensor:
        """
        根据低分辨率路径扩展搜索窗口
        
        参数:
            path: 低分辨率路径 [P, 2]
            radius: 窗口半径
            x_len: 高分辨率序列x的长度
            y_len: 高分辨率序列y的长度
            
        返回:
            window: 高分辨率搜索窗口 [W, 2]，表示(i,j)点的集合
        """
        window_set = set()
        
        # 对路径中的每个点进行扩展
        for idx in range(len(path)):
            # 安全地获取路径中的点坐标
            if isinstance(path, torch.Tensor):
                i, j = path[idx][0].item(), path[idx][1].item()
            else:
                i, j = path[idx][0], path[idx][1]
                
            # FastDTW论文中的投影公式 - 从低分辨率到高分辨率
            i_high = min(i * 2, x_len - 1)
            j_high = min(j * 2, y_len - 1)
            
            # 添加扩展窗口内的所有点
            for i_win in range(max(0, i_high - radius), min(x_len, i_high + radius + 1)):
                for j_win in range(max(0, j_high - radius), min(y_len, j_high + radius + 1)):
                    window_set.add((i_win, j_win))
        
        # 将集合转换为tensor
        window = torch.tensor(list(window_set), dtype=torch.long, 
                              device=path.device if isinstance(path, torch.Tensor) else None)
        
        return window
    
    def _compute_fastdtw(self, 
                        x: torch.Tensor, 
                        y: torch.Tensor, 
                        radius: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        递归计算FastDTW - 修复版本
        
        参数:
            x: 第一个序列 [N, F]
            y: 第二个序列 [M, F]
            radius: 搜索半径
            
        返回:
            distance: DTW距离
            path: 最优对齐路径 [P, 2]
        """
        x_len, y_len = x.shape[0], y.shape[0]
        
        # 基本情况：如果序列足够短，直接使用标准DTW
        if x_len <= 2 or y_len <= 2:
            dist, dtw_matrix = super()._compute_dtw_distance(x, y)
            path = self._extract_optimal_path(dtw_matrix)
            return dist, path
            
        # 下采样
        x_sampled = self._downsample(x)
        y_sampled = self._downsample(y)
        
        # 递归计算低分辨率DTW
        _, low_res_path = self._compute_fastdtw(x_sampled, y_sampled, radius)
        
        # 从低分辨率路径扩展窗口
        window = self._expand_window(low_res_path, radius, x_len, y_len)
        
        # 计算窗口内的欧氏距离矩阵
        dist_mat = torch.full((x_len, y_len), float('inf'), device=x.device)
        
        # 为窗口中的点计算实际距离
        for idx in range(len(window)):
            i, j = window[idx][0].item(), window[idx][1].item()
            dist_mat[i, j] = torch.sqrt(torch.sum((x[i] - y[j]) ** 2))
        
        # 使用约束的距离矩阵计算DTW
        dtw_matrix = torch.full((x_len + 1, y_len + 1), float('inf'), device=x.device)
        dtw_matrix[0, 0] = 0.0
        
        # 确保第一行和第一列有合理的初始值（避免所有值都是inf）
        for i in range(1, x_len + 1):
            if dist_mat[i-1, 0] != float('inf'):
                dtw_matrix[i, 1] = dtw_matrix[i-1, 1] + dist_mat[i-1, 0]
                
        for j in range(1, y_len + 1):
            if dist_mat[0, j-1] != float('inf'):
                dtw_matrix[1, j] = dtw_matrix[1, j-1] + dist_mat[0, j-1]
        
        # 填充DTW矩阵，仅考虑窗口内的点
        for i in range(1, x_len + 1):
            for j in range(1, y_len + 1):
                # 仅在窗口内计算
                if dist_mat[i-1, j-1] != float('inf'):
                    cost = dist_mat[i-1, j-1]
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],
                        dtw_matrix[i, j-1],
                        dtw_matrix[i-1, j-1]
                    )
        
        # 检查是否有有效路径
        if dtw_matrix[x_len, y_len] == float('inf'):
            # 如果窗口太小，没有有效路径，回退到常规DTW
            dist, dtw_matrix = super()._compute_dtw_distance(x, y)
            path = self._extract_optimal_path(dtw_matrix)
            return dist, path
        
        # 提取最优路径
        path = self._extract_optimal_path(dtw_matrix)
        
        return dtw_matrix[x_len, y_len], path
    
    def forward(self, 
               x: torch.Tensor, 
               y: Optional[torch.Tensor] = None, 
               return_path: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        计算FastDTW距离
        
        参数:
            x: 输入序列或序列批次
                - 单序列形状: [N, F]，其中N是序列长度，F是特征维度
                - 批次形状: [B, N, F]，其中B是批次大小
            y: 可选的第二个序列或序列批次
                - 单序列形状: [M, F]
                - 批次形状: [B, M, F]
            return_path: 是否返回最优对齐路径
            
        返回:
            distances: FastDTW距离
                - 单序列对: 标量张量
                - 批次: 形状为 [B] 的张量
            paths: 如果return_path=True，返回最优对齐路径
                - 单序列对: 形状为 [P, 2] 的张量
                - 批次: 形状为 [B] 的列表，每个元素是形状为 [P_i, 2] 的张量
        """
        # 确保输入在正确的设备上
        if self.use_cuda and x.device.type != 'cuda':
            x = x.cuda()
        
        # 处理批次输入
        if len(x.shape) == 3:  # [B, N, F]
            batch_size = x.shape[0]
            
            if y is None:
                # 如果未提供y，计算x中每个序列与自身的DTW距离
                y = x
            elif len(y.shape) == 2:  # [M, F]
                # 如果y是单个序列，将其扩展为批次
                y = y.unsqueeze(0).expand(batch_size, -1, -1)
            
            # 确保y在正确的设备上
            if self.use_cuda and y.device.type != 'cuda':
                y = y.cuda()
            
            # 批量计算FastDTW距离
            distances = torch.zeros(batch_size, device=x.device)
            paths = [] if return_path else None
            
            for i in range(batch_size):
                dist, path = self._compute_fastdtw(x[i], y[i], self.radius)
                distances[i] = dist
                if return_path:
                    paths.append(path)
            
            # 归一化（如果需要）
            if self.normalize:
                for i in range(batch_size):
                    distances[i] = distances[i] / (x[i].shape[0] + y[i].shape[0])
            
            return (distances, paths) if return_path else distances
        
        # 处理单序列输入 [N, F]
        else:
            if y is None:
                y = x
            
            # 确保y在正确的设备上
            if self.use_cuda and y.device.type != 'cuda':
                y = y.cuda()
            
            dist, path = self._compute_fastdtw(x, y, self.radius)
            
            # 归一化（如果需要）
            if self.normalize:
                dist = dist / (x.shape[0] + y.shape[0])
            
            return (dist, path) if return_path else dist


# 使用示例
def example_usage():
    print("开始DTW测试...")
    # 创建两个随机序列 - 降低维度和长度以加速测试
    seq1 = torch.randn(50, 5)  # [50, 5] - 长度为50，特征维度为5的序列
    seq2 = torch.randn(40, 5)  # [40, 5] - 长度为40，特征维度为5的序列
    
    # 批量处理 - 减少批次大小
    batch_size = 4  # 减少批次大小以加速测试
    batch_seq1 = torch.randn(batch_size, 50, 5)  # [4, 50, 5]
    batch_seq2 = torch.randn(batch_size, 40, 5)  # [4, 40, 5]
    
    print("初始化DTW模块...")
    # 初始化DTW模块
    use_gpu = torch.cuda.is_available()
    dtw = DTW(use_cuda=use_gpu, normalize=True)
    
    print("计算单个序列DTW距离...")
    # 计算单个序列的DTW距离
    start_time = time.time()
    distance = dtw(seq1, seq2)
    print(f"单个序列DTW距离: {distance.item()}, 耗时: {time.time() - start_time:.4f}秒")
    
    print("计算带路径的DTW距离...")
    # 计算并返回最优路径
    start_time = time.time()
    distance, path = dtw(seq1, seq2, return_path=True)
    print(f"带路径的DTW距离: {distance.item()}, 路径长度: {len(path)}, 耗时: {time.time() - start_time:.4f}秒")
    
    print("批量计算DTW距离...")
    # 批量计算DTW距离
    start_time = time.time()
    batch_distances = dtw(batch_seq1, batch_seq2)
    print(f"批量DTW平均距离: {batch_distances.mean().item()}, 耗时: {time.time() - start_time:.4f}秒")
    
    print("初始化FastDTW模块...")
    # 使用FastDTW加速计算
    fast_dtw = FastDTW(radius=5, use_cuda=use_gpu, normalize=True)
    
    print("计算单个序列FastDTW距离...")
    # 单个序列FastDTW
    start_time = time.time()
    fast_distance = fast_dtw(seq1, seq2)
    print(f"FastDTW距离: {fast_distance.item()}, 耗时: {time.time() - start_time:.4f}秒")
    
    print("批量计算FastDTW距离...")
    # 批量计算FastDTW距离 - 使用较小的批次
    start_time = time.time()
    batch_fast_distances = fast_dtw(batch_seq1, batch_seq2)
    print(f"批量FastDTW平均距离: {batch_fast_distances.mean().item()}, 耗时: {time.time() - start_time:.4f}秒")
    
    # 验证DTW和FastDTW结果是否接近
    single_diff = abs(distance.item() - fast_distance.item())
    batch_diff = abs(batch_distances.mean().item() - batch_fast_distances.mean().item())
    print(f"单序列DTW与FastDTW差异: {single_diff:.6f}")
    print(f"批量DTW与FastDTW差异: {batch_diff:.6f}")
    print("测试完成!")


if __name__ == "__main__":
    example_usage()