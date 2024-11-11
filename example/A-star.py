import heapq
import matplotlib.pyplot as plt

# 定义节点类
class Node:
    def __init__(self, x, y, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.g = g  # 从起点到当前节点的实际成本
        self.h = h  # 从当前节点到目标的估计成本
        self.f = g + h  # 总成本估计
        self.parent = parent  # 父节点

    def __lt__(self, other):
        return self.f < other.f

# 计算曼哈顿距离作为启发式函数
def heuristic(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)

# 检查移动是否有效
def is_valid_move(x, y, grid):
    if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0:
        return True
    return False

# A*算法主函数
def a_star_search(grid, start, goal):
    open_list = []
    closed_list = set()

    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add((current_node.x, current_node.y))

        if (current_node.x, current_node.y) == (goal_node.x, goal_node.y):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        # 检查四个方向（上下左右）
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            new_x, new_y = current_node.x + dx, current_node.y + dy
            if is_valid_move(new_x, new_y, grid):
                if (new_x, new_y) in closed_list:
                    continue

                new_g = current_node.g + 1
                new_h = heuristic(current_node, goal_node)
                new_node = Node(new_x, new_y, new_g, new_h, current_node)

                heapq.heappush(open_list, new_node)

    return None

# 可视化路径
def visualize_path(grid, path):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='Greys', interpolation='nearest')

    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_y, path_x, color='blue', marker='o', linestyle='-', linewidth=2, markersize=10)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

# 主函数
if __name__ == "__main__":
    # 定义地图，0表示可通过，1表示障碍物
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]

    start = (0, 0)
    goal = (4, 4)

    path = a_star_search(grid, start, goal)

    if path:
        print("找到路径:", path)
        visualize_path(grid, path)
    else:
        print("没有找到路径")