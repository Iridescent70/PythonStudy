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

# 图像尺寸
height, width = image.shape

# 每个格子的大小
cell_height = height // 27
cell_width = width // 38

# 缩进像素数
inset = 3  # 可以根据实际情况调整

# 创建保存数字的文件夹
for i in range(10):
    folder_path = f"src/{i}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# 切割并保存数字
for row in range(27):
    for col in range(38):
        # 计算当前格子的坐标
        x = col * cell_width + 5 * inset
        y = row * cell_height + 2 * inset

        # 调整裁剪范围
        cell = image[y:y + cell_height - 3 * inset, x:x + cell_width - 3 * inset]

        # 确定当前格子中的数字
        if col < 32:  # 0-7
            digit = col // 4
        else:  # 8-9
            digit = (col - 32) // 3 + 8

        # 保存切割后的数字图像
        save_path = f'src/{digit}/{row}_{col}.png'
        cv2.imwrite(save_path, cell)

    print(f"数字 {row} 保存完毕")

print("数字切割完成！")