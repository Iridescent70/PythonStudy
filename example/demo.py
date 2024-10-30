import cv2
import os

# 图像路径
image_path = './4.jpg'

# 检查图像是否存在
if not os.path.exists(image_path):
    print("图片不存在")
    exit()

# 读取图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否读取成功
if image is None:
    print("无法读取图像")
    exit()

# 使用高斯滤波进行降噪
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# 使用Canny进行边缘检测
edges = cv2.Canny(blurred_image, 100, 200)

# 图像尺寸
height, width = edges.shape

# 每个格子的大小
cell_height = height // 27
cell_width = width // 38

# 缩进像素数
inset = 3  # 可以根据实际情况调整

# 创建保存数字的文件夹
output_dir = "src"
os.makedirs(output_dir, exist_ok=True)

for i in range(10):
    folder_path = os.path.join(output_dir, str(i))
    os.makedirs(folder_path, exist_ok=True)

# 切割并保存数字
for row in range(27):
    for col in range(38):
        # 计算当前格子的坐标
        x = col * cell_width + inset
        y = row * cell_height + inset

        # 调整裁剪范围，提取格子内的边缘部分
        cell = edges[y:y + cell_height - 2 * inset, x:x + cell_width - 2 * inset]

        # 确定当前格子中的数字
        if col < 32:  # 0-7
            digit = col // 4
        else:  # 8-9
            digit = 8 + (col - 32) // 3  # 适当修正逻辑

        # 检查数字范围有效性
        if 0 <= digit <= 9:
            # 保存切割后的数字图像
            save_path = os.path.join(output_dir, str(digit), f'{row}_{col}.png')
            cv2.imwrite(save_path, cell)

    print(f"数字 {row} 保存完毕")

print("数字切割完成！")
