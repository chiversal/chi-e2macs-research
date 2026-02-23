import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import deque, defaultdict
import random
import math
import heapq
from abc import ABC, abstractmethod

# ==============
# 根据系统配置中文字体
# ==============
chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name or 'Hei' in f.name or 'YaHei' in f.name or 'Sim' in f.name or 'Noto' in f.name]

if chinese_fonts:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print(f"使用字体: {chinese_fonts[0]}")
else:
    print("警告：未找到中文字体，中文可能显示为方框")


# ==========================================
# 1. 基础组件定义 (严谨性考量)
# ==========================================

class Task:
    """定义AI任务"""

    def __init__(self, task_id, task_type, arrival_time, flops, deadline):
        self.task_id = task_id
        self.type = task_type
        self.arrival_time = arrival_time
        self.flops = flops
        self.deadline = deadline
        # 运行时状态
        self.start_time = -1
        self.finish_time = -1
        self.assigned_unit = None
        self.service_time = 0  # 实际服务时间
        self.wait_time = 0
        self.state = 'pending'  # pending, running, completed

        # 任务特征（用于随机抖动）
        self.compute_intensity = random.uniform(0.8, 1.2)  # 任务自身的计算复杂度变化


class ComputeUnit:
    """定义算力单元 - 增强统计模型"""

    def __init__(self, name, peak_flops, power_tdp, idle_power, switch_latency, memory_bandwidth=25.6):
        self.name = name
        self.peak_flops = peak_flops
        self.power_tdp = power_tdp
        self.idle_power = idle_power  # 空闲功耗
        self.switch_latency = switch_latency  # ms
        self.memory_bandwidth = memory_bandwidth  # GB/s

        # 运行时状态
        self.queue = deque()  # 只存储task对象，不再预存finish_time
        self.busy_until = 0.0  # 记录硬件忙碌的结束时间
        self.current_task = None  # 当前正在执行的任务
        self.idle_start_time = 0.0  # 开始空闲的时间

        # 统计数据
        self.total_service_time = 0.0
        self.total_tasks = 0
        self.total_busy_time = 0.0
        self.total_idle_time = 0.0
        self.last_state_change = 0.0

        # 按任务类型的统计（用于更精确的估算）
        self.service_time_by_type = defaultdict(lambda: {'sum': 0.0, 'sum_square': 0.0, 'count': 0})

        # E2-MACS 历史数据
        self.history_success = defaultdict(lambda: {'success': 0, 'total': 0})

    def get_power_efficiency(self):
        """能效比：FLOPS/W"""
        return self.peak_flops / self.power_tdp

    def get_avg_service_time(self, task_type=None):
        """获取平均服务时间，可按任务类型细分"""
        if task_type and self.service_time_by_type[task_type]['count'] > 0:
            stats = self.service_time_by_type[task_type]
            return stats['sum'] / stats['count']
        # 兜底用全局平均
        return self.total_service_time / max(1, self.total_tasks)

    def get_service_time_moment(self, task_type=None, moment=2):
        """获取服务时间的矩（用于PK公式）"""
        if task_type and self.service_time_by_type[task_type]['count'] > 1:
            stats = self.service_time_by_type[task_type]
            if moment == 1:
                return stats['sum'] / stats['count']
            elif moment == 2:
                return stats['sum_square'] / stats['count']

        # 没有历史数据时，基于理论估算
        base_service = 0.01  # 默认10ms
        if moment == 1:
            return base_service
        else:
            # 假设变异系数为0.3
            return base_service ** 2 * 1.09

    def update_service_time_stats(self, task_type, service_time):
        """更新服务时间统计（一阶矩和二阶矩）"""
        self.service_time_by_type[task_type]['sum'] += service_time
        self.service_time_by_type[task_type]['sum_square'] += service_time ** 2
        self.service_time_by_type[task_type]['count'] += 1
        self.total_service_time += service_time
        self.total_tasks += 1

    def update_busy_idle_time(self, current_time, is_becoming_busy):
        """更新繁忙/空闲时间统计"""
        if is_becoming_busy:
            # 从空闲变为繁忙
            if self.current_task is None:
                idle_duration = current_time - self.last_state_change
                self.total_idle_time += idle_duration
                print(f"    [DEBUG] {self.name} 空闲结束, 空闲时长: {idle_duration * 1000:.2f}ms")
        else:
            # 从繁忙变为空闲
            if self.current_task is not None:
                busy_duration = current_time - self.last_state_change
                self.total_busy_time += busy_duration
                print(f"    [DEBUG] {self.name} 繁忙结束, 繁忙时长: {busy_duration * 1000:.2f}ms, 累计繁忙: {self.total_busy_time * 1000:.2f}ms")
        self.last_state_change = current_time


class SystemState:
    """系统全局状态"""

    def __init__(self, power_budget):
        self.power_budget = power_budget
        self.current_time = 0.0
        self.total_energy = 0.0


# ==========================================
# 2. 核心数学模型 (严谨性考量)
# ==========================================

def calculate_waiting_time(unit, current_time, task_type=None):
    """
    基于M/G/1队列模型估算等待时间。
    考虑：剩余时间 + 队列中任务的预估时间
    """
    wait_time = 0.0

    # 1. 如果当前有任务在执行，计算剩余时间
    if unit.busy_until > current_time:
        wait_time += (unit.busy_until - current_time)

    # 2. 加上队列中已排队任务的预计执行时间
    if len(unit.queue) > 0:
        # 获取该类型任务的平均服务时间
        avg_service_time = unit.get_avg_service_time(task_type)
        wait_time += len(unit.queue) * avg_service_time

    return wait_time


def calculate_pk_waiting_time(lambda_rate, service_time_mean, service_time_second_moment):
    """
    Pollaczek-Khinchin公式计算M/G/1队列的平均等待时间
    Wq = (λ * E[S^2]) / (2 * (1 - ρ))
    这是理论值，用于与仿真对比验证
    """
    if lambda_rate <= 0:
        return 0.0

    # 计算利用率 ρ = λ * E[S]
    rho = lambda_rate * service_time_mean

    if rho >= 1.0:
        return float('inf')  # 系统不稳定

    # Pollaczek-Khinchin 公式
    wq = (lambda_rate * service_time_second_moment) / (2 * (1 - rho))
    return wq


def calculate_critical_rho(service_rate, deadline, variance_factor=1.0):
    """
    计算理论临界利用率
    考虑P95延迟的修正因子
    """
    # 基础M/M/1阈值
    base_rho = 1 - 1 / (service_rate * deadline)

    # 考虑方差的影响（更保守的阈值）
    # 方差越大，临界利用率越低
    adjusted_rho = base_rho / (1 + variance_factor * 0.2)

    return max(0, min(0.95, adjusted_rho))


# ==========================================
# 3. 调度器实现
# ==========================================
class BaseScheduler(ABC):
    def __init__(self, units):
        self.units = units

    @abstractmethod
    def schedule(self, task, current_time, system_state):
        pass

    def update_metrics(self, unit, task, success):
        """更新历史成功率（E2-MACS使用）"""
        pass


# --- 基线策略 ---
class RoundRobinScheduler(BaseScheduler):
    def __init__(self, units):
        super().__init__(units)
        self.idx = 0

    def schedule(self, task, current_time, system_state):
        unit = self.units[self.idx % len(self.units)]
        self.idx += 1
        return unit


class RandomScheduler(BaseScheduler):
    def schedule(self, task, current_time, system_state):
        return random.choice(self.units)


class FastestResponseScheduler(BaseScheduler):
    def schedule(self, task, current_time, system_state):
        # 无脑选峰值算力最高的
        return max(self.units, key=lambda u: u.peak_flops)


class LoadBalanceScheduler(BaseScheduler):
    def schedule(self, task, current_time, system_state):
        # 选队列最短的（考虑当前正在执行的任务）
        def queue_length(unit):
            return len(unit.queue) + (1 if unit.busy_until > current_time else 0)

        return min(self.units, key=queue_length)


class OracleScheduler(BaseScheduler):
    """
    上帝视角调度器（理论最优上界）
    预知未来所有任务到达，但实现简化版：总是选理论上最快的可用单元
    不考虑功耗，只追求最小化延迟
    """

    def schedule(self, task, current_time, system_state):
        # 选择预期完成时间最早的单元
        best_unit = None
        best_finish = float('inf')

        for unit in self.units:
            # 估算在该单元上的完成时间
            compute_time = task.flops / unit.peak_flops * task.compute_intensity
            switch_time = unit.switch_latency / 1000.0
            service_time = compute_time + switch_time

            start_time = max(current_time, unit.busy_until)
            finish_time = start_time + service_time

            if finish_time < best_finish:
                best_finish = finish_time
                best_unit = unit

        return best_unit


# --- E2-MACS 核心逻辑 (严谨性考量) ---
#class E2MACSScheduler(BaseScheduler):


# ==========================================
# 4. 离散事件仿真引擎 (严谨性考量)
# ==========================================

class SimulationEngine:
    def __init__(self, scheduler_class, config):
        self.scheduler_class = scheduler_class
        self.config = config
        self.units = []
        self.scheduler = None
        self.reset()

    def reset(self):
        # 初始化硬件（加入空闲功耗）
        cfg = self.config['hardware']
        self.units = [
            ComputeUnit("CPU", cfg['cpu_flops'], cfg['cpu_power'],
                        cfg['cpu_idle_power'], cfg['cpu_latency']),
            ComputeUnit("GPU", cfg['gpu_flops'], cfg['gpu_power'],
                        cfg['gpu_idle_power'], cfg['gpu_latency']),
            ComputeUnit("NPU", cfg['npu_flops'], cfg['npu_power'],
                        cfg['npu_idle_power'], cfg['npu_latency'])
        ]
        self.scheduler = self.scheduler_class(self.units)
        self.current_time = 0.0
        self.event_queue = []  # 优先队列 (time, type, obj)
        self.completed_tasks = []
        self.task_id_counter = 0
        self.tasks = []  # 所有任务
        self.total_energy = 0.0
        self.last_energy_update = 0.0

        # 初始化各单元的状态变更时间
        for unit in self.units:
            unit.last_state_change = 0.0

    def _update_energy(self, current_time):
        """更新能耗累计"""
        dt = current_time - self.last_energy_update
        if dt <= 0:
            return

        for unit in self.units:
            if unit.current_task is not None:
                # 繁忙状态：TDP功耗
                power = unit.power_tdp
            else:
                # 空闲状态：空闲功耗
                power = unit.idle_power
            self.total_energy += power * dt

        self.last_energy_update = current_time

    def _generate_task_arrivals(self):
        """生成泊松到达过程（支持非均匀任务类型分布）"""
        duration = self.config['sim_duration']
        rate = self.config['arrival_rate']
        task_dist = self.config.get('task_distribution', [0.33, 0.33, 0.34])

        t = 0
        while t < duration:
            # 泊松过程间隔
            dt = np.random.exponential(1.0 / rate)
            t += dt
            if t >= duration:
                break

            # 根据分布选择任务类型
            r = random.random()
            if r < task_dist[0]:
                t_type, flops, deadline = 'voice', 10e6, 0.05
            elif r < task_dist[0] + task_dist[1]:
                t_type, flops, deadline = 'image', 100e6, 0.2
            else:
                t_type, flops, deadline = 'detect', 500e6, 0.3

            task = Task(self.task_id_counter, t_type, t, flops, deadline)
            self.task_id_counter += 1
            self.tasks.append(task)
            # 插入到达事件
            heapq.heappush(self.event_queue, (t, 'arrival', task))

    def _calculate_service_time(self, task, unit, current_time):
        """
        计算实际服务时间（带随机抖动）
        这是任务实际执行的时间，用于仿真
        """
        # 基础计算时间
        base_compute = task.flops / unit.peak_flops

        # 引入随机波动（模拟真实系统的性能抖动）
        # 使用对数正态分布，更符合实际系统的长尾特性
        noise = np.random.lognormal(0, 0.1)  # 均值≈1，方差可控
        compute_time = base_compute * noise * task.compute_intensity

        # 内存带宽约束（也加入随机性）
        memory_time = task.flops * 4 / (unit.memory_bandwidth * 1e9)
        memory_noise = np.random.lognormal(0, 0.05)
        effective_memory = memory_time * memory_noise

        effective_compute = max(compute_time, effective_memory)

        # 切换开销（固定）
        switch_time = unit.switch_latency / 1000.0

        return effective_compute + switch_time

    def _process_arrival(self, task):
        """处理任务到达事件"""
        # 更新能耗
        self._update_energy(self.current_time)

        # 调度决策
        target_unit = self.scheduler.schedule(task, self.current_time, None)

        # 计算服务时间（实际执行时间）
        service_time = self._calculate_service_time(task, target_unit, self.current_time)

        # 确定开始时间
        start_time = max(self.current_time, target_unit.busy_until)
        finish_time = start_time + service_time

        # 记录任务信息
        task.start_time = start_time
        task.assigned_unit = target_unit.name
        task.service_time = service_time
        task.wait_time = start_time - self.current_time
        task.state = 'running' if start_time == self.current_time else 'pending'

        # 更新单元统计（服务时间统计）- 入队时先记录，完成时再确认
        target_unit.update_service_time_stats(task.type, service_time)

        # 更新单元状态
        if start_time == self.current_time:
            # 立即开始执行
            if target_unit.current_task is None:
                # 从空闲变为繁忙
                target_unit.update_busy_idle_time(self.current_time, True)

            target_unit.current_task = task
            target_unit.busy_until = finish_time
            # 插入完成事件
            heapq.heappush(self.event_queue, (finish_time, 'finish', (task, target_unit)))
        else:
            # 加入队列（只存task对象，不预存finish_time）
            target_unit.queue.append(task)
            # 注意：不需要更新busy_until，因为当前任务还没开始

    def _process_finish(self, task, unit):
        """处理任务完成事件"""
        # 更新能耗
        self._update_energy(self.current_time)

        task.finish_time = self.current_time
        task.state = 'completed'

        # 计算是否成功（满足截止时间）
        success = (task.finish_time - task.arrival_time) <= task.deadline

        # 更新调度器的历史指标
        self.scheduler.update_metrics(unit, task, success)

        # 记录完成任务
        self.completed_tasks.append(task)

        # 更新单元状态 - 当前任务完成
        unit.update_busy_idle_time(self.current_time, False)  # 变为空闲
        unit.current_task = None

        # 检查队列中是否有等待的任务
        if unit.queue:
            # 取出下一个任务（FIFO）
            next_task = unit.queue.popleft()

            # 重新计算服务时间（基于当前状态）
            next_service_time = self._calculate_service_time(next_task, unit, self.current_time)
            next_finish_time = self.current_time + next_service_time

            # 更新任务
            next_task.start_time = self.current_time
            next_task.state = 'running'

            # 更新单元状态 - 变为繁忙
            unit.current_task = next_task
            unit.busy_until = next_finish_time
            unit.update_busy_idle_time(self.current_time, True)  # 变为繁忙

            # 插入新的完成事件
            heapq.heappush(self.event_queue, (next_finish_time, 'finish', (next_task, unit)))

    def run(self):
        """运行仿真"""
        self.reset()
        self._generate_task_arrivals()
        self.last_energy_update = 0.0

        # 仿真主循环
        while self.event_queue:
            time, event_type, obj = heapq.heappop(self.event_queue)
            self.current_time = time

            if event_type == 'arrival':
                self._process_arrival(obj)
            elif event_type == 'finish':
                task, unit = obj
                self._process_finish(task, unit)

        # 最后更新一次能耗
        self._update_energy(self.config['sim_duration'])

        return self._calc_metrics()

    def _calc_metrics(self):
        """计算性能指标"""
        if not self.completed_tasks:
            return {
                'strategy': self.scheduler.__class__.__name__,
                'avg_latency': 0,
                'p95_latency': 0,
                'p99_latency': 0,
                'success_rate': 0,
                'npu_load': 0,
                'total_energy': 0,
                'throughput': 0,
                'avg_wait_time': 0,
                'total_tasks': 0,
                'rho_critical_theory': 0,
                'rho_critical_observed': 0
            }

        df = pd.DataFrame([{
            'task_id': t.task_id,
            'type': t.type,
            'arrival': t.arrival_time,
            'start': t.start_time,
            'finish': t.finish_time,
            'assigned': t.assigned_unit,
            'service': t.service_time,
            'wait': t.wait_time,
            'deadline': t.deadline,
            'compute_intensity': t.compute_intensity
        } for t in self.completed_tasks])

        df['latency'] = df['finish'] - df['arrival']
        df['success'] = df['latency'] <= df['deadline']

        # 任务分配统计
        unit_counts = {}
        type_distribution = {}
        for t in self.completed_tasks:
            # 按单元统计
            unit_counts[t.assigned_unit] = unit_counts.get(t.assigned_unit, 0) + 1
            # 按类型统计
            key = f"{t.type}->{t.assigned_unit}"
            type_distribution[key] = type_distribution.get(key, 0) + 1

        print(f"\n  [分配统计] {self.scheduler.__class__.__name__}:")
        print(f"    单元分布: {unit_counts}")
        print(f"    类型-单元分布: {type_distribution}")
        # 计算各单元利用率
        sim_duration = self.config['sim_duration']
        for unit in self.units:
            unit.utilization = unit.total_busy_time / sim_duration

        # 计算NPU负载
        npu_unit = next(u for u in self.units if u.name == 'NPU')
        npu_rho = npu_unit.utilization

        # 计算理论ρ_critical（使用P95修正）
        avg_task_flops = np.mean([10e6, 100e6, 500e6])
        mu_npu = self.config['hardware']['npu_flops'] / avg_task_flops

        # 获取NPU上任务的服务时间二阶矩（用于方差估计）
        if npu_unit.service_time_by_type:
            all_tasks = []
            for t_type in npu_unit.service_time_by_type:
                stats = npu_unit.service_time_by_type[t_type]
                if stats['count'] > 0:
                    all_tasks.extend([stats['sum'] / stats['count']] * stats['count'])

            if all_tasks:
                service_times = np.array(all_tasks)
                cv = np.std(service_times) / np.mean(service_times)  # 变异系数
            else:
                cv = 0.3  # 默认值
        else:
            cv = 0.3

        D_avg = np.mean([0.05, 0.2, 0.3])
        rho_critical_theory = calculate_critical_rho(mu_npu, D_avg, cv)

        # 观察到的崩溃点（P95延迟超过截止时间）
        p95_latency = df['latency'].quantile(0.95)
        p95_exceed_deadline = p95_latency > D_avg

        # 能量消耗（已在仿真过程中累计）
        total_energy = self.total_energy

        # 吞吐量（任务/秒）
        throughput = len(self.completed_tasks) / sim_duration

        return {
            'strategy': self.scheduler.__class__.__name__,
            'avg_latency': df['latency'].mean() * 1000,  # 转换为ms
            'p95_latency': df['latency'].quantile(0.95) * 1000,
            'p99_latency': df['latency'].quantile(0.99) * 1000,
            'success_rate': df['success'].mean(),
            'npu_load': npu_rho,
            'rho_critical_theory': rho_critical_theory,
            'rho_critical_observed': npu_rho if p95_exceed_deadline else 0,
            'total_energy': total_energy,
            'throughput': throughput,
            'avg_wait_time': df['wait'].mean() * 1000,
            'total_tasks': len(self.completed_tasks)
        }


# ==========================================
# 5. 实验运行与可视化
# ==========================================

def run_single_experiment(config, strategies):
    """运行单次实验，所有策略使用相同的随机种子"""
    results = []

    print(f"\n{'=' * 60}")
    print(f"实验配置: 到达率 λ={config['arrival_rate']} 任务/秒")
    print(f"任务分布: {config.get('task_distribution', [0.33, 0.33, 0.34])}")
    print(f"{'=' * 60}")

    for Strat in strategies:
        # 固定随机种子，确保公平对比
        random.seed(42)
        np.random.seed(42)

        engine = SimulationEngine(Strat, config)
        stats = engine.run()
        results.append(stats)

        # 打印结果
        print(f"\n{stats['strategy']}:")
        print(f"  成功率: {stats['success_rate'] * 100:.2f}%")
        print(f"  平均延迟: {stats['avg_latency']:.2f} ms")
        print(f"  P95延迟: {stats['p95_latency']:.2f} ms")
        print(f"  P99延迟: {stats['p99_latency']:.2f} ms")
        print(f"  NPU负载: {stats['npu_load']:.3f}")
        print(f"  吞吐量: {stats['throughput']:.2f} 任务/秒")
        print(f"  总能耗: {stats['total_energy']:.2f} J")

    return results


def run_lambda_sweep(config, strategies, lambda_range):
    """扫描不同到达率"""
    sweep_results = []

    for lam in lambda_range:
        config['arrival_rate'] = lam
        for Strat in strategies:
            random.seed(42)
            np.random.seed(42)

            engine = SimulationEngine(Strat, config)
            stats = engine.run()
            stats['lambda'] = lam
            sweep_results.append(stats)

        print(f"完成 λ={lam} 扫描")

    return pd.DataFrame(sweep_results)


def plot_results(results_df):
    """绘制结果对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 成功率对比
    ax = axes[0, 0]
    for strategy in results_df['strategy'].unique():
        data = results_df[results_df['strategy'] == strategy]
        ax.plot(data['lambda'], data['success_rate'] * 100, marker='o', linewidth=2, label=strategy)
    ax.set_xlabel('到达率 λ (任务/秒)')
    ax.set_ylabel('成功率 (%)')
    ax.set_title('成功率 vs 负载')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95%目标')

    # 2. 平均延迟对比
    ax = axes[0, 1]
    for strategy in results_df['strategy'].unique():
        data = results_df[results_df['strategy'] == strategy]
        ax.plot(data['lambda'], data['avg_latency'], marker='o', linewidth=2, label=strategy)
    ax.set_xlabel('到达率 λ (任务/秒)')
    ax.set_ylabel('平均延迟 (ms)')
    ax.set_title('平均延迟 vs 负载')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. P95延迟对比（关键指标）
    ax = axes[0, 2]
    for strategy in results_df['strategy'].unique():
        data = results_df[results_df['strategy'] == strategy]
        ax.plot(data['lambda'], data['p95_latency'], marker='o', linewidth=2, label=strategy)
    ax.set_xlabel('到达率 λ (任务/秒)')
    ax.set_ylabel('P95延迟 (ms)')
    ax.set_title('P95延迟 vs 负载 (关键QoS指标)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # 标注平均截止时间
    ax.axhline(y=200, color='red', linestyle='--', alpha=0.5, label='典型截止时间(200ms)')

    # 4. NPU负载对比
    ax = axes[1, 0]
    for strategy in results_df['strategy'].unique():
        data = results_df[results_df['strategy'] == strategy]
        ax.plot(data['lambda'], data['npu_load'], marker='o', linewidth=2, label=strategy)

    # 标注理论临界值
    if 'rho_critical_theory' in results_df.columns:
        theory_rho = results_df['rho_critical_theory'].iloc[0]
        ax.axhline(y=theory_rho, color='red', linestyle='--', alpha=0.5,
                   label=f'理论ρ_critical({theory_rho:.2f})')
    ax.axhline(y=0.7, color='orange', linestyle=':', alpha=0.5, label='ρ=0.7(关注点)')
    ax.axhline(y=0.9, color='red', linestyle=':', alpha=0.5, label='ρ=0.9(拥塞)')

    ax.set_xlabel('到达率 λ (任务/秒)')
    ax.set_ylabel('NPU负载 ρ')
    ax.set_title('NPU利用率 vs 负载')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 吞吐量对比
    ax = axes[1, 1]
    for strategy in results_df['strategy'].unique():
        data = results_df[results_df['strategy'] == strategy]
        ax.plot(data['lambda'], data['throughput'], marker='o', linewidth=2, label=strategy)
    ax.plot(data['lambda'], data['lambda'], 'k--', alpha=0.5, label='理想吞吐量')
    ax.set_xlabel('到达率 λ (任务/秒)')
    ax.set_ylabel('吞吐量 (任务/秒)')
    ax.set_title('吞吐量 vs 负载')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 能耗对比
    ax = axes[1, 2]
    for strategy in results_df['strategy'].unique():
        data = results_df[results_df['strategy'] == strategy]
        ax.plot(data['lambda'], data['total_energy'], marker='o', linewidth=2, label=strategy)
    ax.set_xlabel('到达率 λ (任务/秒)')
    ax.set_ylabel('总能耗 (J)')
    ax.set_title('能耗 vs 负载')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e2macs_simulation_results.png', dpi=150)
    plt.show()


def main():
    # 基础配置（加入空闲功耗）
    CONFIG = {
        'hardware': {
            'cpu_flops': 10e9, 'cpu_power': 2.0, 'cpu_idle_power': 0.5, 'cpu_latency': 1,
            'gpu_flops': 50e9, 'gpu_power': 5.0, 'gpu_idle_power': 1.0, 'gpu_latency': 5,
            'npu_flops': 100e9, 'npu_power': 3.0, 'npu_idle_power': 0.8, 'npu_latency': 10
        },
        'sim_duration': 200,  # 仿真时长（秒），增加以提高统计显著性
        'task_distribution': [0.33, 0.33, 0.34]  # 均匀分布
    }

    strategies = [
        RandomScheduler,
        RoundRobinScheduler,
        LoadBalanceScheduler,
        FastestResponseScheduler,
        E2MACSScheduler,
        OracleScheduler  # 加入上帝视角作为上界
    ]

    # 实验1：单点测试（中等负载）
    print("\n=== 实验1: 中等负载测试 (λ=10) ===")
    CONFIG['arrival_rate'] = 10
    results = run_single_experiment(CONFIG, strategies)

    # 实验2：负载扫描
    print("\n=== 实验2: 负载扫描测试 ===")
    lambda_range = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100]
    sweep_df = run_lambda_sweep(CONFIG, strategies, lambda_range)

    # 保存结果
    sweep_df.to_csv('simulation_results.csv', index=False)
    print(f"\n结果已保存到 simulation_results.csv")

    # 绘制图表
    plot_results(sweep_df)

    # 实验3：非均匀任务分布测试
    print("\n=== 实验3: 非均匀任务分布 (检测任务占60%) ===")
    CONFIG['task_distribution'] = [0.2, 0.2, 0.6]  # 60%检测任务
    CONFIG['arrival_rate'] = 8  # 降低负载避免过早崩溃
    results_heavy = run_single_experiment(CONFIG, [FastestResponseScheduler, E2MACSScheduler, OracleScheduler])

    # 验证ρ_critical
    print("\n=== 理论验证 ===")
    # 获取最后一次运行的NPU数据
    npu_unit = None
    for r in results_heavy:
        if r['strategy'] == 'E2MACSScheduler':
            print(f"E2-MACS 在重负载场景下的表现:")
            print(f"  成功率: {r['success_rate'] * 100:.2f}%")
            print(f"  P95延迟: {r['p95_latency']:.2f} ms")
            print(f"  NPU负载: {r['npu_load']:.3f}")
            print(f"  理论ρ_critical: {r['rho_critical_theory']:.3f}")


def main_small():
    # 基础配置
    CONFIG = {
        'hardware': {
            'cpu_flops': 10e9, 'cpu_power': 2.0, 'cpu_idle_power': 0.5, 'cpu_latency': 1,
            'gpu_flops': 50e9, 'gpu_power': 5.0, 'gpu_idle_power': 1.0, 'gpu_latency': 5,
            'npu_flops': 100e9, 'npu_power': 3.0, 'npu_idle_power': 0.8, 'npu_latency': 10
        },
        'sim_duration': 50,  # 缩短仿真时间，快速看到结果
        'task_distribution': [0.33, 0.33, 0.34]
    }

    # 只测试两个关键策略：FastestResponse 和 E2MACSScheduler
    strategies = [
        FastestResponseScheduler,
        E2MACSScheduler,
    ]

    # 只测试一个中等负载
    print("\n=== 测试分配情况 (λ=10) ===")
    CONFIG['arrival_rate'] = 10

    for Strat in strategies:
        random.seed(42)
        np.random.seed(42)

        engine = SimulationEngine(Strat, CONFIG)
        stats = engine.run()

        print(f"\n{stats['strategy']} 最终统计:")
        print(f"  成功率: {stats['success_rate'] * 100:.2f}%")
        print(f"  NPU负载: {stats['npu_load']:.3f}")


def main_small_10s():
    CONFIG = {
        'hardware': {
            'cpu_flops': 10e9, 'cpu_power': 2.0, 'cpu_idle_power': 0.5, 'cpu_latency': 1,
            'gpu_flops': 50e9, 'gpu_power': 5.0, 'gpu_idle_power': 1.0, 'gpu_latency': 5,
            'npu_flops': 100e9, 'npu_power': 3.0, 'npu_idle_power': 0.8, 'npu_latency': 10
        },
        'sim_duration': 10,  # 缩短到10秒，减少输出
        'task_distribution': [0.33, 0.33, 0.34]
    }

    # 只测试 FastestResponseScheduler 一个策略，避免输出太多
    strategies = [FastestResponseScheduler]

    print("\n=== 测试繁忙时间统计 (λ=10) ===")
    CONFIG['arrival_rate'] = 10

    for Strat in strategies:
        random.seed(42)
        np.random.seed(42)

        engine = SimulationEngine(Strat, CONFIG)
        stats = engine.run()

        print(f"\n{stats['strategy']} 最终统计:")
        print(f"  成功率: {stats['success_rate'] * 100:.2f}%")
        print(f"  NPU负载: {stats['npu_load']:.3f}")
        print(f"  完成任务数: {stats['total_tasks']}")

        # 额外输出各单元的繁忙时间
        for unit in engine.units:
            print(f"  {unit.name} 累计繁忙: {unit.total_busy_time * 1000:.2f}ms, 利用率: {unit.total_busy_time / CONFIG['sim_duration']:.3f}")


def main2():
    # 基础配置（加入空闲功耗）
    CONFIG = {
        'hardware': {
            'cpu_flops': 10e9, 'cpu_power': 2.0, 'cpu_idle_power': 0.5, 'cpu_latency': 1,
            'gpu_flops': 50e9, 'gpu_power': 5.0, 'gpu_idle_power': 1.0, 'gpu_latency': 5,
            'npu_flops': 100e9, 'npu_power': 3.0, 'npu_idle_power': 0.8, 'npu_latency': 10
        },
        'sim_duration': 200,  # 仿真时长（秒），增加以提高统计显著性
        'task_distribution': [0.33, 0.33, 0.34]  # 均匀分布
    }

    strategies = [
        RandomScheduler,
        RoundRobinScheduler,
        LoadBalanceScheduler,
        FastestResponseScheduler,
        E2MACSScheduler,
        OracleScheduler  # 加入上帝视角作为上界
    ]

    # 实验1：单点测试（中等负载）
    print("\n=== 实验1: 中等负载测试 (λ=10) ===")
    CONFIG['arrival_rate'] = 10
    results = run_single_experiment(CONFIG, strategies)

    # 实验2：负载扫描
    print("\n=== 实验2: 负载扫描测试 ===")
    lambda_range = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    sweep_df = run_lambda_sweep(CONFIG, strategies, lambda_range)
    sweep_df['distribution'] = 'uniform'  # 标记为均匀分布

    # 保存结果
    sweep_df.to_csv('simulation_results.csv', index=False)
    print(f"\n结果已保存到 simulation_results.csv")

    # 绘制图表
    plot_results(sweep_df)

    # ========== 新增代码开始 ==========

    # 实验3：非均匀任务分布测试 - 也做负载扫描
    print("\n=== 实验3: 非均匀任务分布 (检测任务占60%) 负载扫描 ===")
    CONFIG['task_distribution'] = [0.2, 0.2, 0.6]  # 60%检测任务

    # 对于重负载场景，使用较低的负载范围避免过早崩溃
    heavy_lambda_range = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    heavy_sweep_df = run_lambda_sweep(CONFIG, strategies, heavy_lambda_range)
    heavy_sweep_df['distribution'] = 'heavy_detect'  # 标记为检测任务占优

    # 保存非均匀分布结果
    heavy_sweep_df.to_csv('simulation_results_heavy.csv', index=False)
    print(f"\n非均匀分布结果已保存到 simulation_results_heavy.csv")

    # 合并两个分布的结果用于对比
    combined_df = pd.concat([sweep_df, heavy_sweep_df], ignore_index=True)
    combined_df.to_csv('simulation_results_combined.csv', index=False)
    print(f"合并结果已保存到 simulation_results_combined.csv")

    # 绘制两种分布的对比图
    plot_distribution_comparison(combined_df)

    # 原有的单点测试保留，但改为从扫描结果中提取数据
    print("\n=== 实验4: 非均匀分布单点验证 (来自扫描数据) ===")
    heavy_at_8 = heavy_sweep_df[(heavy_sweep_df['lambda'] == 8) &
                                (heavy_sweep_df['strategy'].isin(['FastestResponseScheduler', 'E2MACSScheduler', 'OracleScheduler']))]

    for _, r in heavy_at_8.iterrows():
        print(f"\n{r['strategy']} at λ=8:")
        print(f"  成功率: {r['success_rate'] * 100:.2f}%")
        print(f"  P95延迟: {r['p95_latency']:.2f} ms")
        print(f"  NPU负载: {r['npu_load']:.3f}")
        print(f"  理论ρ_critical: {r['rho_critical_theory']:.3f}")

    # 验证ρ_critical - 观察两种分布下的差异
    print("\n=== 理论验证: 均匀分布 vs 非均匀分布 ===")

    # 找到均匀分布下 FastestResponse 开始崩溃的点
    uniform_fast = sweep_df[sweep_df['strategy'] == 'FastestResponseScheduler']
    crash_lambda_uniform = None
    for i in range(1, len(uniform_fast)):
        if uniform_fast.iloc[i]['success_rate'] < uniform_fast.iloc[i - 1]['success_rate'] * 0.8:
            crash_lambda_uniform = uniform_fast.iloc[i]['lambda']
            break

    # 找到非均匀分布下 FastestResponse 开始崩溃的点
    heavy_fast = heavy_sweep_df[heavy_sweep_df['strategy'] == 'FastestResponseScheduler']
    crash_lambda_heavy = None
    for i in range(1, len(heavy_fast)):
        if heavy_fast.iloc[i]['success_rate'] < heavy_fast.iloc[i - 1]['success_rate'] * 0.8:
            crash_lambda_heavy = heavy_fast.iloc[i]['lambda']
            break

    print(f"均匀分布下 FastestResponse 崩溃点: λ={crash_lambda_uniform}")
    print(f"非均匀分布下 FastestResponse 崩溃点: λ={crash_lambda_heavy}")

    # 在λ=10时对比两种分布下 E2-MACS 的表现
    compare_at_10 = combined_df[(combined_df['lambda'] == 10) &
                                (combined_df['strategy'] == 'E2MACSScheduler')]

    for _, r in compare_at_10.iterrows():
        print(f"\nE2-MACS at λ=10, {r['distribution']} 分布:")
        print(f"  成功率: {r['success_rate'] * 100:.2f}%")
        print(f"  P95延迟: {r['p95_latency']:.2f} ms")
        print(f"  NPU负载: {r['npu_load']:.3f}")

    # ========== 新增代码结束 ==========


# ========== 新增函数：绘制两种分布的对比图 ==========
def plot_distribution_comparison(df):
    """绘制均匀分布 vs 非均匀分布的对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    distributions = df['distribution'].unique()
    colors = {'uniform': 'blue', 'heavy_detect': 'red'}
    markers = {'uniform': 'o', 'heavy_detect': 's'}

    metrics = [
        ('success_rate', '成功率 (%)', lambda x: x * 100),
        ('avg_latency', '平均延迟 (ms)', lambda x: x),
        ('p95_latency', 'P95延迟 (ms)', lambda x: x),
        ('npu_load', 'NPU负载 ρ', lambda x: x),
        ('total_energy', '总能耗 (J)', lambda x: x),
        ('throughput', '吞吐量 (任务/秒)', lambda x: x)
    ]

    for idx, (metric, ylabel, transform) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        for dist in distributions:
            dist_df = df[df['distribution'] == dist]

            # 画 E2-MACS 和 FastestResponse 的对比
            for strategy in ['E2MACSScheduler', 'FastestResponseScheduler']:
                data = dist_df[dist_df['strategy'] == strategy]
                if not data.empty:
                    y = transform(data[metric])

                    label = f"{dist}-{strategy}"
                    linestyle = '-' if strategy == 'E2MACSScheduler' else '--'

                    ax.plot(data['lambda'], y,
                            marker=markers[dist],
                            color=colors[dist],
                            linestyle=linestyle,
                            linewidth=2,
                            label=label)

        ax.set_xlabel('到达率 λ (任务/秒)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} 对比')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

        # 对成功率图添加参考线
        if metric == 'success_rate':
            ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5, label='95%目标')
        # 对延迟图添加截止时间参考线
        elif metric == 'p95_latency':
            ax.axhline(y=200, color='red', linestyle=':', alpha=0.5, label='截止时间(200ms)')

    plt.tight_layout()
    plt.savefig('distribution_comparison.png', dpi=150)
    plt.show()


def plot_all_distributions(uniform_df, heavy_df, voice_df):
    """分别绘制三种分布的对比图"""

    # 图1：均匀分布
    plt.figure(figsize=(15, 10))
    plot_single_distribution(uniform_df, '均匀分布')
    plt.savefig('uniform_distribution.png', dpi=150)
    plt.show()

    # 图2：检测占优分布
    plt.figure(figsize=(15, 10))
    plot_single_distribution(heavy_df, '检测任务占优 (60%)')
    plt.savefig('heavy_detect_distribution.png', dpi=150)
    plt.show()

    # 图3：语音占优分布
    plt.figure(figsize=(15, 10))
    plot_single_distribution(voice_df, '语音任务占优 (60%)')
    plt.savefig('heavy_voice_distribution.png', dpi=150)
    plt.show()

    # 图4：三种分布对比（只画E2-MACS）
    plt.figure(figsize=(15, 10))
    plot_e2macs_comparison(uniform_df, heavy_df, voice_df)
    plt.savefig('e2macs_comparison.png', dpi=150)
    plt.show()


def plot_single_distribution(df, title):
    """绘制单个分布的六子图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    strategies = ['FastestResponseScheduler', 'LoadBalanceScheduler', 'E2MACSScheduler', 'OracleScheduler']
    colors = {'FastestResponseScheduler': 'red',
              'LoadBalanceScheduler': 'orange',
              'E2MACSScheduler': 'green',
              'OracleScheduler': 'blue'}
    markers = {'FastestResponseScheduler': 'o',
               'LoadBalanceScheduler': 's',
               'E2MACSScheduler': '^',
               'OracleScheduler': 'd'}

    metrics = [
        ('success_rate', '成功率 (%)', lambda x: x * 100, 0, 105),
        ('p95_latency', 'P95延迟 (ms)', lambda x: x, 0, None),
        ('npu_load', 'NPU负载 ρ', lambda x: x, 0, 1),
        ('total_energy', '总能耗 (J)', lambda x: x, 0, None),
        ('throughput', '吞吐量 (任务/秒)', lambda x: x, 0, None),
        ('avg_latency', '平均延迟 (ms)', lambda x: x, 0, None)
    ]

    for idx, (metric, ylabel, transform, ymin, ymax) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        for strategy in strategies:
            data = df[df['strategy'] == strategy]
            if not data.empty:
                y = transform(data[metric])
                ax.plot(data['lambda'], y,
                        marker=markers[strategy],
                        color=colors[strategy],
                        linewidth=2,
                        label=strategy.replace('Scheduler', ''))

        ax.set_xlabel('到达率 λ (任务/秒)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs 负载')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        if ymin is not None:
            ax.set_ylim(ymin=ymin)
        if ymax is not None:
            ax.set_ylim(ymax=ymax)

        # 对延迟图添加截止时间参考线
        if metric == 'p95_latency':
            ax.axhline(y=200, color='gray', linestyle=':', alpha=0.7, label='截止时间(200ms)')
        # 对成功率图添加95%参考线
        elif metric == 'success_rate':
            ax.axhline(y=95, color='gray', linestyle=':', alpha=0.7, label='95%目标')

    plt.suptitle(f'调度策略对比 - {title}', fontsize=16)
    plt.tight_layout()


def plot_e2macs_comparison(uniform_df, heavy_df, voice_df):
    """对比E2-MACS在三种分布下的表现"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    distributions = [
        (uniform_df, '均匀分布', 'blue', 'o'),
        (heavy_df, '检测占优(60%)', 'red', 's'),
        (voice_df, '语音占优(60%)', 'green', '^')
    ]

    metrics = [
        ('success_rate', '成功率 (%)', lambda x: x * 100, 0, 105),
        ('p95_latency', 'P95延迟 (ms)', lambda x: x, 0, None),
        ('npu_load', 'NPU负载 ρ', lambda x: x, 0, 1),
        ('total_energy', '总能耗 (J)', lambda x: x, 0, None),
        ('throughput', '吞吐量 (任务/秒)', lambda x: x, 0, None),
        ('avg_latency', '平均延迟 (ms)', lambda x: x, 0, None)
    ]

    for idx, (metric, ylabel, transform, ymin, ymax) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        for dist_df, dist_name, color, marker in distributions:
            e2_data = dist_df[dist_df['strategy'] == 'E2MACSScheduler']
            if not e2_data.empty:
                y = transform(e2_data[metric])
                ax.plot(e2_data['lambda'], y,
                        marker=marker,
                        color=color,
                        linewidth=2,
                        label=dist_name)

        ax.set_xlabel('到达率 λ (任务/秒)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'E2-MACS {ylabel} 对比')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        if ymin is not None:
            ax.set_ylim(ymin=ymin)
        if ymax is not None:
            ax.set_ylim(ymax=ymax)

        if metric == 'p95_latency':
            ax.axhline(y=200, color='gray', linestyle=':', alpha=0.7, label='截止时间')

    plt.suptitle('E2-MACS 在不同任务分布下的表现对比', fontsize=16)
    plt.tight_layout()


def main3():
    # 基础配置
    CONFIG = {
        'hardware': {
            'cpu_flops': 10e9, 'cpu_power': 2.0, 'cpu_idle_power': 0.5, 'cpu_latency': 1,
            'gpu_flops': 50e9, 'gpu_power': 5.0, 'gpu_idle_power': 1.0, 'gpu_latency': 5,
            'npu_flops': 100e9, 'npu_power': 3.0, 'npu_idle_power': 0.8, 'npu_latency': 10
        },
        'sim_duration': 200,
        'task_distribution': [0.33, 0.33, 0.34]  # 默认均匀分布
    }

    strategies = [
        RandomScheduler,
        RoundRobinScheduler,
        LoadBalanceScheduler,
        FastestResponseScheduler,
        E2MACSScheduler,
        OracleScheduler
    ]

    # === 实验1：均匀分布负载扫描 ===
    print("\n=== 实验1: 均匀分布负载扫描 ===")
    CONFIG['task_distribution'] = [0.33, 0.33, 0.34]
    lambda_range_1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    uniform_df = run_lambda_sweep(CONFIG, strategies, lambda_range_1)
    uniform_df['distribution'] = 'uniform'
    uniform_df.to_csv('simulation_results_uniform.csv', index=False)

    # === 实验2：检测任务占优 (60%检测) ===
    print("\n=== 实验2: 检测任务占优 (60%检测) 负载扫描 ===")
    CONFIG['task_distribution'] = [0.2, 0.2, 0.6]  # 60%检测任务
    lambda_range_2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]  # 检测任务多，负载能力低
    heavy_df = run_lambda_sweep(CONFIG, strategies, lambda_range_2)
    heavy_df['distribution'] = 'heavy_detect'
    heavy_df.to_csv('simulation_results_heavy.csv', index=False)

    # === 实验3：语音任务占优 (60%语音) - 新增高识别场景 ===
    print("\n=== 实验3: 语音任务占优 (60%语音) 负载扫描 ===")
    CONFIG['task_distribution'] = [0.6, 0.2, 0.2]  # 60%语音任务
    lambda_range_3 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    # 语音任务小，可以跑到更高负载
    voice_df = run_lambda_sweep(CONFIG, strategies, lambda_range_3)
    voice_df['distribution'] = 'heavy_voice'
    voice_df.to_csv('simulation_results_voice.csv', index=False)

    # === 合并所有结果 ===
    combined_df = pd.concat([uniform_df, heavy_df, voice_df], ignore_index=True)
    combined_df.to_csv('simulation_results_complete.csv', index=False)
    print(f"\n完整结果已保存到 simulation_results_complete.csv")

    # === 绘制三种分布对比图 ===
    plot_three_distributions(combined_df)

    # === 打印关键对比数据 ===
    print("\n=== 三种任务分布在 λ=20 时的对比 ===")

    # 提取 λ=20 的数据
    lambda_20_data = combined_df[combined_df['lambda'] == 20]

    for dist in ['uniform', 'heavy_detect', 'heavy_voice']:
        dist_data = lambda_20_data[lambda_20_data['distribution'] == dist]
        if dist_data.empty:
            continue

        print(f"\n[{dist}] 分布:")
        for strategy in ['FastestResponseScheduler', 'LoadBalanceScheduler', 'E2MACSScheduler', 'OracleScheduler']:
            row = dist_data[dist_data['strategy'] == strategy]
            if not row.empty:
                r = row.iloc[0]
                print(f"  {strategy}: 成功率={r['success_rate'] * 100:.1f}%, "
                      f"P95={r['p95_latency']:.1f}ms, "
                      f"NPU负载={r['npu_load']:.3f}, "
                      f"能耗={r['total_energy']:.0f}J")
    # 读取三个分布的数据文件（如果还没有在内存中）
    uniform_df = pd.read_csv('simulation_results_uniform.csv')
    heavy_df = pd.read_csv('simulation_results_heavy.csv')
    voice_df = pd.read_csv('simulation_results_voice.csv')

    # 生成所有图表
    plot_all_distributions(uniform_df, heavy_df, voice_df)

    # 打印关键对比数据
    print("\n=== 三种分布在 λ=20 时的关键对比 ===")
    lambda_20_data = []
    for df, name in [(uniform_df, '均匀分布'), (heavy_df, '检测占优'), (voice_df, '语音占优')]:
        e2_data = df[(df['strategy'] == 'E2MACSScheduler') & (df['lambda'] == 20)]
        if not e2_data.empty:
            r = e2_data.iloc[0]
            print(f"\n{name}:")
            print(f"  成功率: {r['success_rate'] * 100:.2f}%")
            print(f"  P95延迟: {r['p95_latency']:.2f} ms")
            print(f"  NPU负载: {r['npu_load']:.3f}")
            print(f"  总能耗: {r['total_energy']:.2f} J")


def plot_three_distributions(df):
    """绘制三种分布下的对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    distributions = df['distribution'].unique()
    colors = {'uniform': 'blue', 'heavy_detect': 'red', 'heavy_voice': 'green'}
    markers = {'uniform': 'o', 'heavy_detect': 's', 'heavy_voice': '^'}

    metrics = [
        ('success_rate', '成功率 (%)', lambda x: x * 100),
        ('p95_latency', 'P95延迟 (ms)', lambda x: x),
        ('npu_load', 'NPU负载 ρ', lambda x: x),
        ('total_energy', '总能耗 (J)', lambda x: x),
        ('throughput', '吞吐量 (任务/秒)', lambda x: x),
        ('avg_latency', '平均延迟 (ms)', lambda x: x)
    ]

    for idx, (metric, ylabel, transform) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        for dist in distributions:
            dist_df = df[df['distribution'] == dist]

            # 只画 E2-MACS 和 FastestResponse
            for strategy in ['E2MACSScheduler', 'FastestResponseScheduler']:
                data = dist_df[dist_df['strategy'] == strategy]
                if not data.empty:
                    y = transform(data[metric])
                    label = f"{dist}-{strategy}"
                    linestyle = '-' if strategy == 'E2MACSScheduler' else '--'

                    ax.plot(data['lambda'], y,
                            marker=markers[dist],
                            color=colors[dist],
                            linestyle=linestyle,
                            linewidth=2,
                            label=label)

        ax.set_xlabel('到达率 λ (任务/秒)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} 对比 (三种分布)')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        # 对成功率图添加参考线
        if metric == 'success_rate':
            ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5, label='95%目标')
        # 对延迟图添加截止时间参考线
        elif metric == 'p95_latency':
            ax.axhline(y=200, color='red', linestyle=':', alpha=0.5, label='截止时间(200ms)')

    plt.tight_layout()
    plt.savefig('three_distributions_comparison.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()  # λ跨度更大，包含100极端场景只绘制了任务均匀分布，实验3只是打印结果
    # main_small() # Debug 没有NPU负载数据
    # main_small_10s() # Debug 没有NPU负载数据 -- 根因：数据在记录前被设置成了None
    main3()
