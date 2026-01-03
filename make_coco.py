# 导入必要的库
import os  # 操作系统接口
import pickle
import numpy as np  # 数值计算库

# 将json标注文件的信息和ID进行映射
def make_id_dict(jsonData: dict, location:str,id:str,contents:str):
    id_dict = {}  # 字典，用于存储映射关系

    # 遍历数据中的每一项
    for item in jsonData[location]:
        # 提取索引键和索引值
        key = item[id]  # 索引键，图像ID
        value = item[contents]  # 索引值，映射内容

        # 如果键不存在，创建新条目
        if key not in id_dict:
            id_dict.update({key: [value]})  # 创建列表存储第一个值
        else:
            # 如果键已存在，将值添加到现有列表中
            id_dict[key].append(value)

    return id_dict  # 返回处理结果


# ============ 处理数据 ============
def process(PATH,dataset):
    # 读取JSON标注文件
    jsonFile = os.path.join(PATH, "annotations", f"captions_{dataset}2017.json")
    with open(jsonFile, "r") as f:
        jsonData = json.load(f)
    # 创建ID-索引字典
    indexDict = make_id_dict(jsonData, "images","id","file_name")
    # 创建ID-描述字典
    captionDict = make_id_dict(jsonData, "annotations","image_id","caption")

    # 读取JSON标注文件
    jsonFile = os.path.join(PATH, "annotations", f"instances_{dataset}2017.json")
    with open(jsonFile, "r") as f:
        jsonData = json.load(f)
    # 创建ID-类别字典
    categoryDict = make_id_dict(jsonData, "annotations","image_id","category_id")

    # ============ 找出共有的ID（确保数据对齐） ============
    # 获取三个字典中都存在的图像ID
    index_ids = set(indexDict.keys())
    caption_ids = set(captionDict.keys())
    category_ids = set(categoryDict.keys())
    common_ids = index_ids & caption_ids & category_ids  # 交集
    print(f"index:{len(index_ids)}、caption:{len(caption_ids)}、category:{len(category_ids)},有{len(common_ids)}个完整样本")
    # 只保留三个字典中都有的图像ID，确保数据对齐
    indexDict = {img_id: indexDict[img_id] for img_id in common_ids}
    captionDict = {img_id: captionDict[img_id] for img_id in common_ids}
    categoryDict = {img_id: categoryDict[img_id] for img_id in common_ids}

    # ============ 将ID-类别字典转为ID-独热编码字典 ============
    # 读取JSON标注文件
    jsonFile = os.path.join(PATH, "annotations", f"instances_{dataset}2017.json")
    with open(jsonFile, "r") as f:
        jsonData = json.load(f)
    # 创建类别ID-类别索引字典【1~90=>0~类别长度】
    reflectDict = {}
    for i,category in enumerate(jsonData["categories"]):
        reflectDict[category["id"]] = i

    # ============ 按ID排序并存储为列表 ============
    indexList = []
    captionList = []
    categoryList = []
    # 将ID转换为整数排序，确保一致的顺序
    sorted_ids = sorted(common_ids, key=lambda x: int(x) if str(x).isdigit() else x)
    # 创建索引、描述和类别的列表
    for id in sorted_ids:
        # 获取索引
        indexList.append(f"{dataset}2017/"+indexDict[id][0])
        # 获取描述
        captionList.append(captionDict[id])
        # 获取类别【类别ID列表转为多标签独热编码】
        code = np.zeros(len(reflectDict), dtype=np.int8)
        for category in categoryDict[id]:
            code[reflectDict[category]] = 1
        categoryList.append(code)

    return indexList, captionList, categoryList

# 主程序入口
if __name__ == "__main__":
    import json  # JSON处理库
    import scipy.io as scio  # MATLAB文件读写库
    import argparse  # 命令行参数解析库

    # 创建命令行参数解析器.在字符串前加 r 表示原始字符串
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", default=r"D:\BaiduNetdiskDownload\clip-hash-dataset.tar\clip-hash-dataset\baidu-clip-hash-dataset\coco\coco2017", type=str,
                        help="COCO数据集目录路径")
    parser.add_argument("--save-dir", default="./pkl_dataset", type=str,
                        help="PKL文件保存目录")
    args = parser.parse_args()  # 解析命令行参数

    # 设置路径
    PATH = args.coco_dir  # COCO数据集根目录

    # 可以验证，ID和文件名是一一对应的，139==>000000000139.jpg
    # jsonFile = os.path.join(PATH, "annotations", f"captions_train2017.json")
    # with open(jsonFile, "r") as f:
    #     jsonData = json.load(f)
    # for item in jsonData["images"]:
    #     filename = int(item["file_name"].split(".")[0])
    #     if filename != item["id"]:
    #         print("并不总相等",filename, item["id"])
    #         break
    # exit()
    

    indexList,captionList,categoryList = process(PATH, "train")
    print(f"val数据集大小: 图像数量={len(indexList)}, 描述数量={sum(len(sublist) for sublist in captionList)}, 类别={len(categoryList[0])}")
    print(indexList[0:2])
    print(captionList[0:2])
    print(categoryList[0:2])
    indexList1, captionList1, categoryList1 = process(PATH, "val")
    # 将验证集数据追加到训练集数据后面
    indexList.extend(indexList1)  # 合并图像文件名
    captionList.extend(captionList1)  # 合并描述
    categoryList.extend(categoryList1)  # 合并类别标签

    print(f"最终数据集大小: 图像数量={len(indexList)}, 描述数量={sum(len(sublist) for sublist in captionList)}, 类别={len(categoryList[0])}")

    # exit()

    # ============ 保存为pkl格式文件 ============
    # 把Python对象（如列表、字典、numpy数组等）转换成二进制格式并保存到文件中。
    # 准备保存的数据结构
    data_dict = {"indexs": indexList,  # 图片索引
                 "captions": captionList,  # 文本描述
                 "labels": categoryList}  # 标签矩阵

    # 保存为.pkl文件（pikle格式）
    # ============ 创建 mat 目录（如果不存在） ============
    pkl_dir = "pkl_dataset"
    os.makedirs(pkl_dir, exist_ok=True)

    #  保存为pickle格式（Python原生，加载最快）
    # "wb"参数：w=写入模式，b=二进制模式（pickle需要二进制）
    # pickle.dump()函数：将Python对象序列化并写入文件
    # 函数语法：pickle.dump(要保存的对象, 文件对象, 协议版本)
    with open(os.path.join(pkl_dir, "coco2017.pkl"), "wb") as f:
        pickle.dump(data_dict, f)

    print(f"finished!see {pkl_dir}")  # 完成提示

# D:\Anaconda3\envs\study\pythonw.exe C:/Users/dy/Desktop/CMR_BASE/dataset/make_minicoco.py
# index:118287、caption:118287、category:117266,有117266个完整样本
# val数据集大小: 图像数量=117266, 描述数量=586646, 类别=80
# ['train2017/000000000009.jpg', 'train2017/000000000025.jpg']
# [['Closeup of bins of food that include broccoli and bread.', 'A meal is presented in brightly colored plastic trays.', 'there are containers filled with different kinds of foods', 'Colorful dishes holding meat, vegetables, fruit, and bread.', 'A bunch of trays that have different food.'], ['A giraffe eating food from the top of the tree.', 'A giraffe standing up nearby a tree ', 'A giraffe mother with its baby in the forest.', 'Two giraffes standing in a tree filled area.', 'A giraffe standing next to a forest filled with trees.']]
# [array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int8), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int8)]
# index:5000、caption:5000、category:4952,有4952个完整样本
# 最终数据集大小: 图像数量=122218, 描述数量=611420, 类别=80
# finished!see pkl_dataset
#
# 进程已结束,退出代码0