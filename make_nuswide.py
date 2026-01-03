# 导入必要的库
import os  # 操作系统接口
import numpy as np  # 数值计算库
import pickle # 导入pickle模块 - Python内置的序列化模块


# 设置NUS-WIDE数据集的根目录
# 需要修改为实际的下载目录路径
root_dir = "raw_dataset/nuswide" # 数据集根路径

# 定义数据文件路径
imageListFile = os.path.join(root_dir, "ImageList/Imagelist.txt") # 图片路径索引
labelPath = os.path.join(root_dir, "Groundtruth/AllLabels") # 81个类别标签文件【每个文件存储n行0-1】
textFile = os.path.join(root_dir, "NUS_WID_Tags/All_Tags.txt") # 图像ID、文本描述(多词)

# 真实图像文件夹路径
imagePath = "images/Flickr"

# ============ 1. 加载图像索引 ============
with open(imageListFile, "r") as f:
    indexs = f.readlines()  # 读取所有行，每行是一个图像路径。返回一个列表，列表中的每个元素是文件的一行（字符串）每行末尾包含换行符 \n

# 处理图像路径：去除换行符，将反斜杠\替换为正斜杠/, 添加完整路径前缀
indexs = [os.path.join(imagePath, item.strip()).replace("\\", "/") for item in indexs]
print("indexs length:", len(indexs))  # 打印图像数量
print(indexs[0:2])
# [img1,img2...img269648]



# ============ 2. 加载文本描述 ============
captions = []  # 存储文本描述的列表
with open(textFile, "r",encoding="utf-8") as f:
    for line in f:
        if len(line.strip()) == 0:  # 跳过空行
            print("some line empty!")
            continue

        # 处理文本行：第一列可能是索引，后面是标签词
        caption = line.split()[1:]  # 跳过第一列（图像ID）
        caption = " ".join(caption).strip()  # 用空格连接标签词

        if len(caption) == 0:  # 如果描述为空，使用占位符
            caption = "123456"  # 占位文本

        captions.append(caption)  # 添加到列表

print("captions length:", len(captions))  # 打印描述数量
print(captions[0:2])
# [text1,text2...text269648]


# ============ 3. 加载标签（类别）信息 ============
# 加载使用的标签列表（NUS-WIDE常用的类别）【从81个类别中选取常用的类别】
with open(os.path.join(root_dir, "ConceptsList/Concepts81_sort.txt")) as f:
    label_lists = f.readlines()  # 读取标签文件的所有行,返回一个列表，列表中的每个元素是文件的一行（字符串）每行末尾包含换行符 \n

label_lists = [item.strip() for item in label_lists[0:21]]  # 只取常见类别，去除换行符
# ["cat","dog",..."class21"]

# 创建类别索引映射（标签名 -> 索引位置）
class_index = {}
for i, item in enumerate(label_lists):
    class_index.update({item: i})  # 映射关系，如："animal" -> 0
# {"cat":0,"dog":1,..."class21":20}

# 创建标签矩阵：形状为[图像数量, 类别数量]，数据类型为int8（节省内存）
labels = np.zeros([len(indexs), len(class_index)], dtype=np.int8)
# [0,0,...0]
# [0,0,...0]
#  ........
# [0,0,...0]

# 通过类别标签文件填充编码矩阵
for item in label_lists:
    path = os.path.join(labelPath, "Labels_"+item+".txt")  # 标签文件路径【n行0-1】
    class_label = item  # 类别名称

    with open(path, "r") as f:
        data = f.readlines()  # 读取标签文件的所有行【1行对应一个图像】

    # 为每个图像设置该类别标签
    for i, val in enumerate(data):
        # 如果值为"1"，表示该图像属于该类别【多标签独热编码】
        labels[i][class_index[class_label]] = 1 if val.strip() == "1" else 0

print("labels sum:", labels.sum())  # 打印所有标签的总和（统计正样本数量）
print(labels[0:2])
# [[0,1,...0],[],...,[]]


# ============ 4. 过滤全 0 标签的图像（推荐方式） ============
# labels: [num_images, num_classes]

# 1. 计算每个样本是否至少有一个标签
valid_mask = labels.sum(axis=1) > 0    # shape: [num_images]
# [true,false,.......]

print("before filtering:")
print("indexs length:", len(indexs))
print("captions length:", len(captions))
print("labels shape:", labels.shape)

# 2. 根据 mask 过滤 indexs、captions、labels【只保留True(在选定类别中)的部分】
indexs = [idx for idx, keep in zip(indexs, valid_mask) if keep]
captions = [cap for cap, keep in zip(captions, valid_mask) if keep]
labels = labels[valid_mask]

# 等价于
# new_indexs = []
# new_captions = []
# for i in range(len(labels)):
#     if labels[i].sum() > 0:
#         new_indexs.append(indexs[i])
#         new_captions.append(captions[i])
# indexs = new_indexs
# captions = new_captions
# labels = labels[labels.sum(axis=1) > 0]

print("after filtering:")
print("indexs length:", len(indexs)) # 打印过滤后的图像数量
print("captions length:", len(captions)) # 打印过滤后的描述数量
print("labels shape:", labels.shape) # 打印标签矩阵的形状
print("labels sum:", labels.sum()) #打印标签矩阵有多少个1【多标签】


print("\n===== 每个类别的图像数量 =====")
for i, class_name in enumerate(label_lists):
    num = labels[:, i].sum()
    print(f"{class_name:20s}: {num}")



# ============ 5. 保存为pkl格式文件 ============
# 把Python对象（如列表、字典、numpy数组等）转换成二进制格式并保存到文件中。
# 准备保存的数据结构
data_dict = {"indexs": indexs, # 图片索引
               "captions": captions, # 文本描述
               "labels": labels}  # 标签矩阵


# 保存为.pkl文件（pikle格式）
# ============ 创建 mat 目录（如果不存在） ============
pkl_dir = "pkl_dataset"
os.makedirs(pkl_dir, exist_ok=True)

#  保存为pickle格式（Python原生，加载最快）
# "wb"参数：w=写入模式，b=二进制模式（pickle需要二进制）
# pickle.dump()函数：将Python对象序列化并写入文件
# 函数语法：pickle.dump(要保存的对象, 文件对象, 协议版本)
with open(os.path.join(pkl_dir, "nuswide.pkl"), "wb") as f:
    pickle.dump(data_dict, f)


print(f"finished!see {pkl_dir}")  # 完成提示