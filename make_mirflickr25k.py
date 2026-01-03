import os # 导入操作系统接口模块，用于处理文件和目录路径
import numpy as np # 导入NumPy库，用于数值计算和数组操作
import pickle # 导入pickle模块 - Python内置的序列化模块
# 数据预处理脚本：将MIRFlickr-25K数据集转换为pkl格式

# 设置数据集根目录，需要替换为实际下载目录
root_dir = "raw_dataset/mirflickr25k"

# 构建标签文件路径
# 对应MIRFlickr-25K数据集的标注文件目录
file_path = os.path.join(root_dir, "mirflickr25k_annotations_v080")

# 获取标签目录下的所有文件列表
file_list = os.listdir(file_path)

# 过滤文件列表，移除包含"_r1"的文件和README文件
# "_r1"可能是重复文件，README是说明文件
file_list = [item for item in file_list if "_r1" not in item and "README" not in item]

# 打印类别数量（每个文件对应一个类别）
print("class num:", len(file_list))

# 创建类别索引字典：类别文件名 -> 索引编号
class_index = {}
# enumerate返回(索引, 元素)对，i从0开始递增
for i, item in enumerate(file_list):
    class_index.update({item: i})  # 将类别文件名映射到数字索引


# 创建标签字典：{图像ID : 多标签向量}【不是所有图片都在给定类别当中】
label_dict = {}
# 遍历每个类别文件
for path_id in file_list:
    # 构建完整的文件路径
    path = os.path.join(file_path, path_id)
    # 打开类别文件读取
    with open(path, "r") as f:
        # 逐行读取文件内容
        for item in f:
            item = item.strip()  # 移除字符串两端的空白字符
            # 将读取的字符串转为int
            item = int(item)
            # 如果图像ID不在字典中，创建新的标签向量
            if item not in label_dict:
                # 创建全零向量，长度为类别总数
                label = np.zeros(len(file_list),dtype=np.int8)
                # 将当前类别对应的位置设为1（one-hot编码）
                label[class_index[path_id]] = 1
                # 添加到字典：图像ID -> 标签向量
                label_dict.update({item: label})
            else:
                # 如果图像ID已存在，更新对应类别的标签为1
                # 说明该图像属于多个类别（多标签）
                label_dict[item][class_index[path_id]] = 1

# 打印至少有一个类别的图像数量
print("sample size:", len(label_dict))
# 打印丢弃的样本
miss = set(range(1,25001)) - set(label_dict.keys())
print("miss",len(miss))
print(f"miss {sorted(list(miss))[0:3]} ...")

# 获取所有图像ID并排序，确保顺序一致
keys = list(label_dict.keys())
keys.sort()  # 按数字顺序排序


# 创建标签矩阵：按排序后的图像ID顺序提取标签向量
labels = []
for key in keys:
    labels.append(label_dict[key])  # 添加对应图像的标签向量
print("labels created:", len(labels))

# 构建图像文件路径列表
PATH = "mirflickr/"
# 为每个图像ID构建完整的.jpg文件路径
# 图像命名格式：im{图像ID}.jpg
indexs = [PATH + "im" + str(item) + ".jpg" for item in keys]
print("index created:", len(indexs))


# 处理文本描述（captions）
captions_path = os.path.join(root_dir, "mirflickr/meta/tags")
# 获取所有标签文件（每个图像对应一个标签文件）
captions_list = os.listdir(captions_path)
# 创建描述字典：图像ID -> 文本描述
captions_dict = {}
# 遍历每个标签文件
for item in captions_list:
    # 从文件名提取图像ID：tags12345.txt -> 12345
    id_ = item.split(".")[0].replace("tags", "")
    id_ = int(id_)
    caption = ""  # 初始化空描述字符串
    # 打开文件读取内容
    with open(os.path.join(captions_path, item), "r",encoding="utf-8") as f:
        # 读取所有行，每行是一个标签词
        for word in f.readlines():
            caption += word.strip() + " "  # 将标签词用空格连接
    caption = caption.strip()  # 移除首尾空格
    # 添加到字典：图像ID -> 文本描述
    captions_dict.update({id_: caption})

# 创建描述列表：按排序后的图像ID顺序提取描述
captions = []
for item in keys:
    # 每个描述包装成列表
    captions.append([captions_dict[item]])

print("captions created:", len(captions))


# ============ 保存为pkl格式文件 ============
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
with open(os.path.join(pkl_dir, "flickr25k.pkl"), "wb") as f:
    pickle.dump(data_dict, f)


print(f"finished!see {pkl_dir}")  # 完成提示





