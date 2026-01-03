# 导入必要的库
import os  # 操作系统接口
import pickle

import numpy as np  # 数值计算库


# 将json标注文件的字典提取有效信息再得到需要的字典
def make_index(jsonData: dict, indexDict: dict):
    """
    从json标注文件的字典中提取有效信息再得到需要的字典，合并为列表

    参数:
    jsonData: COCO JSON格式的字典数据
    indexDict: 例如{
        "images": ["id", "file_name"],  # 图像ID -> 文件名
        "annotations": ["image_id", "caption"]  # 图像ID -> 描述文本
    }

    返回:
    result: 重新组织后的字典列表，例如 [
        {"id1": ["file_name1"],...},  # 图像ID -> 文件名
        {"id1": ["text1","text2","text3","text4","text5"],...}  # 图像ID -> 描述文本
    ]

    """
    result = []  # 存储处理结果的列表

    # 遍历indexDict中的每个键
    for name in indexDict:
        data = jsonData[name]  # 获取JSON数据中对应的部分（如所有图像信息）
        middle_dict = {}  # 中间字典，用于存储映射关系

        # 遍历数据中的每一项
        for item in data:
            # 提取索引键和索引值
            key = item[indexDict[name][0]]  # 索引键，图像ID
            value = item[indexDict[name][1]]  # 索引值，图像文件名、描述文本

            # 如果键不存在，创建新条目
            if key not in middle_dict:
                middle_dict.update({key: [value]})  # 创建列表存储第一个值
            else:
                # 如果键已存在，将值添加到现有列表中
                middle_dict[key].append(value)

        result.append(middle_dict)  # 将处理结果添加到结果列表

    return result  # 返回处理结果


# 检查文件是否存在的函数
def check_file_exist(indexDict: dict, file_path: str, prefix: str = ""):
    """
    检查索引字典中的文件路径是否存在

    参数:
    indexDict: 索引字典，键为标识符，值为文件路径
    file_path: 文件基础路径

    返回:
    indexDict: 更新后的索引字典，只包含实际存在的文件

    功能说明:
    检查每个文件路径是否存在，如果不存在则从字典中移除该条目
    """
    keys = list(indexDict.keys())  # 获取字典的所有键

    # 遍历所有键
    for item in keys:
        # 检查文件是否存在
        if not os.path.exists(os.path.join(file_path, indexDict[item][0])):
            print(f"文件不存在: {item}, {indexDict[item]}")  # 打印警告信息
            indexDict.pop(item)  # 移除不存在的文件条目
        else:
            # 如果文件存在，更新路径为完整路径
            indexDict[item] = prefix + indexDict[item][0]

    return indexDict  # 返回更新后的字典


# 将类别id转换为80维的独热编码
def chage_categories2numpy(category_ids: dict, data: dict):
    """
    将类别字典转换为numpy数组（多标签one-hot编码）

    参数:
    category_ids: 类别ID到索引的映射字典【【1~80 => 0~79】】
    data: 包含类别信息的字典，键为图像ID，值为类别ID列表

    返回:
    data: 更新后的字典，类别信息转换为numpy数组

    功能说明:
    将每个图像的类别列表转换为one-hot编码的numpy数组
    """
    # 遍历字典中的每个图像ID
    for item in data:
        # 创建初始为0的类别向量
        class_item = [0] * len(category_ids)  # 长度等于类别总数

        # 遍历该图像的所有类别ID
        for class_id in data[item]:
            # 将对应的类别位置设为1
            class_item[category_ids[class_id]] = 1

        # 将类别ID列表转换为numpy数组
        data[item] = np.asarray(class_item)

    return data  # 返回更新后的字典


# 获取所有使用的键的函数
def get_all_use_key(categoryDict: dict):
    """
    获取字典中所有可用的键

    参数:
    categoryDict: 类别字典

    返回:
    list: 字典中所有键的列表
    """
    return list(categoryDict.keys())  # 返回字典的所有键


# 移除未使用的键
def remove_not_use(data: dict, used_key: list):
    """
    从字典中移除未使用的键

    参数:
    data: 要清理的字典
    used_key: 应该保留的键列表

    返回:
    data: 清理后的字典
    """
    keys = list(data.keys())  # 获取当前字典的所有键

    # 遍历所有键
    for item in keys:
        # 如果键不在使用列表中，移除它
        if item not in used_key:
            # print(f"移除: {item}, {indexDict[item]}")  # 调试信息
            data.pop(item)  # 移除未使用的键

    return data  # 返回清理后的字典


# 将字典的值按ID升序存放为列表
def merge_to_list(data: dict):
    """
    将字典按键排序后合并为列表

    参数:
    data: 输入字典

    返回:
    result: 排序后的值列表

    功能说明:
    按键排序，确保顺序一致，然后提取值到列表中
    """
    result = []  # 结果列表
    key_sort = list(data.keys())  # 获取所有键
    key_sort.sort()  # 按键排序（升序）

    # 按排序后的键顺序提取值
    for item in key_sort:
        result.append(data[item])  # 添加对应的值

    return result  # 返回排序后的值列表


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

    # ============ 处理训练集数据 ============

    # 1. 处理训练集的描述信息
    jsonFile = os.path.join(PATH, "annotations", "captions_train2017.json")
    with open(jsonFile, "r") as f:
        jsonData = json.load(f)  # 加载JSON文件

    # 定义要提取的信息
    indexDict = {
        "images": ["id", "file_name"],  # 图像ID -> 文件名
        "annotations": ["image_id", "caption"]  # 图像ID -> 描述文本
    }

    # 提取图片路径索引和描述信息
    result = make_index(jsonData, indexDict)
    indexDict_, captionDict = result  # 解包结果
    '''
        indexDict_ = {"id1": ["file_name1"], ...},  # 图像ID -> 文件名
        captionDict = {"id1": ["text1", "text2", "text3", "text4", "text5"], ...}  # 图像ID -> 描述文本
    '''

    # 检查图像文件是否存在，打印数据集信息
    indexDict_ = check_file_exist(indexDict_, os.path.join(PATH, "train2017"),"train2017/")
    print(f"train2017/: 图像数量={len(indexDict_)}, 描述数量={sum(len(sublist) for sublist in captionDict.values())}")
    '''
            indexDict_ = {"id1": "file_name1", ...},  # 图像ID -> 文件名
    '''

    # 2. 处理训练集的类别标签信息
    jsonFile = os.path.join(PATH, "annotations", "instances_train2017.json")
    with open(jsonFile, "r") as f:
        jsonData = json.load(f)

    # 创建类别ID到索引的映射【1~90 => 0~89】
    categroy_ids = {}
    for i, item in enumerate(jsonData['categories']):
        categroy_ids.update({item['id']: i})  # 类别ID -> 索引位置

    # 定义要提取的索引（类别信息）
    indexDict = {
        "annotations": ["image_id", "category_id"],  # 图像ID -> 类别ID
        "images": ["id", "file_name"]  # 图像ID -> 文件名
    }

    # 提取类别和图像信息
    result = make_index(jsonData, indexDict)
    categoryDict = result[0]  # 图像ID -> 类别ID列表
    cateIndexDict = result[1]  # 图像ID -> 文件名

    # 将类别信息转换为numpy数组（one-hot编码）
    categoryDict = chage_categories2numpy(categroy_ids, categoryDict)
    '''
        categoryDict = {"id1": [0,1,0,0,1,1,0,...1,0], ...}  # 图像ID -> 类别标签
        cateIndexDict = {"id1": ["file_name1"], ...},  # 图像ID -> 文件名
    '''


    # 3. 统一所有数据的键（确保数据对齐）【将能完整成对的图片-文本-标签保留下来，过滤掉没有描述或其他情况的数据】
    used_key = get_all_use_key(categoryDict)  # 获取类别字典中的所有键【图像ID】

    # 清理其他字典，只保留与类别字典相同的键
    indexDict_ = remove_not_use(indexDict_, used_key)  # 清理图像索引
    captionDict = remove_not_use(captionDict, used_key)  # 清理描述
    categoryIndexDict = remove_not_use(cateIndexDict, used_key)  # 清理图像文件名
    categoryDict = remove_not_use(categoryDict, used_key)  # 清理类别

    '''  【处理后id是统一的】
        indexDict_ = {"id1": "file_name1", ...},  # 图像ID -> 文件名
        captionDict = {"id1": ["text1", "text2", "text3", "text4", "text5"], ...}  # 图像ID -> 描述文本
        cateIndexDict = {"id1": ["file_name1"], ...},  # 图像ID -> 文件名
        categoryDict = {"id1": [0,1,0,0,1,1,0,...1,0], ...}  # 图像ID -> 类别标签
    '''

    # 4. 将所有字典转换为列表（保持顺序一致）
    indexList = merge_to_list(indexDict_)  # 图像文件名列表
    captionList = merge_to_list(captionDict)  # 图像描述列表
    categoryIndexList = merge_to_list(categoryIndexDict)  # 验证用的图像文件名列表
    categoryList = merge_to_list(categoryDict)  # 类别标签列表
    '''  【按图像id升序排列】
        indexList = ["file_name1", ...],  # 文件名
        captionList = [["text1", "text2", "text3", "text4", "text5"], ...]  # 文本描述
        categoryIndexList = [["file_name1"], ...],  # 文件名
        categoryList = [[0,1,0,0,1,1,0,...1,0], ...]  # 类别标签
    '''

    print(f"处理后train2017/: 图像数量={len(indexDict_)}, 描述数量={sum(len(sublist) for sublist in captionList)}, 类别={len(categoryList[0])}")

    # 5. 验证数据对齐（确保索引一致）【确保两个json文件的相同图像ID对应的是同一张图片】【一般都是对的，不然就出事了】
    for i in range(len(indexList)):
        if i < 10:  # 只打印前10行
            if indexList[i] != "train2017/" + categoryIndexList[i][0]:
                print(f"不一致: 索引={i}, 描述文件对应的图像文件名={indexList[i]}, 类别文件对应的图像文件名={categoryIndexList[i]}")

    # ============ 处理验证集数据（与训练集类似） ============

    # 1. 处理验证集的描述信息
    val_jsonFile = os.path.join(PATH, "annotations", "captions_val2017.json")
    with open(val_jsonFile, "r") as f:
        jsonData = json.load(f)

    indexDict = {
        "images": ["id", "file_name"],
        "annotations": ["image_id", "caption"]
    }

    result = make_index(jsonData, indexDict)
    val_indexDict = result[0]  # 验证集图像索引
    val_captionDict = result[1]  # 验证集描述

    val_indexDict = check_file_exist(val_indexDict, os.path.join(PATH, "val2017"),"val2017/")
    print(f"val2017/: 图像数量={len(val_indexDict)}, 描述数量={sum(len(sublist) for sublist in val_captionDict.values())}")

    # 2. 处理验证集的类别信息
    jsonFile = os.path.join(PATH, "annotations", "instances_val2017.json")
    with open(jsonFile, "r") as f:
        jsonData = json.load(f)

    # 创建验证集的类别映射
    categroy_ids = {}
    for i, item in enumerate(jsonData['categories']):
        categroy_ids.update({item['id']: i})

    indexDict = {
        "annotations": ["image_id", "category_id"],
        "images": ["id", "file_name"]
    }

    result = make_index(jsonData, indexDict)
    val_categoryDict = result[0]  # 验证集类别
    val_categoryIndexDict = result[1]  # 验证集图像索引

    # 转换为numpy数组
    val_categoryDict = chage_categories2numpy(categroy_ids, val_categoryDict)

    # 3. 统一验证集的键
    used_key = get_all_use_key(val_categoryDict)
    val_indexDict = remove_not_use(val_indexDict, used_key)
    val_captionDict = remove_not_use(val_captionDict, used_key)
    val_categoryIndexDict = remove_not_use(val_categoryIndexDict, used_key)
    val_categoryDict = remove_not_use(val_categoryDict, used_key)

    # 4. 将验证集字典转换为列表
    val_indexList = merge_to_list(val_indexDict)
    val_captionList = merge_to_list(val_captionDict)
    val_categoryIndexList = merge_to_list(val_categoryIndexDict)
    val_categoryList = merge_to_list(val_categoryDict)

    # 5. 验证数据对齐（确保索引一致）【确保两个json文件的相同图像ID对应的是同一张图片】【一般都是对的，不然就出事了】
    for i in range(len(val_indexList)):
        if i < 10: # 只打印前10行
            if val_indexList[i] != "val2017/" + val_categoryIndexList[i][0]:
                print(f"不一致: 索引={i}, 描述文件对应的图像文件名={val_indexList[i]}, 类别文件对应的图像文件名={val_categoryIndexList[i]}")

    print(f"处理后val2017/: 图像数量={len(val_indexDict)}, 描述数量={sum(len(sublist) for sublist in val_captionList)}, 类别={len(val_categoryList[0])}")

    # ============ 合并训练集和验证集，得到一个大的数据集(后续可自由划分) ============

    # 将验证集数据追加到训练集数据后面
    indexList.extend(val_indexList)  # 合并图像文件名
    captionList.extend(val_captionList)  # 合并描述
    categoryIndexList.extend(val_categoryIndexList)  # 合并图像索引（验证用）【这里其实后续用不上，路径索引留一个就行了】
    categoryList.extend(val_categoryList)  # 合并类别标签


    print(f"最终数据集大小: 图像数量={len(indexList)}, 描述数量={sum(len(sublist) for sublist in captionList)}, 类别={len(categoryList[0])}")


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
    with open(os.path.join(pkl_dir, "coco2017_old.pkl"), "wb") as f:
        pickle.dump(data_dict, f)

    print(f"finished!see {pkl_dir}")  # 完成提示

    # D:\Anaconda3\envs\study\pythonw.exe
    # C: / Users / dy / Desktop / CMR_BASE / dataset / make_coco.py
    # train2017 /: 图像数量 = 118287, 描述数量 = 591753
    # 处理后train2017 /: 图像数量 = 117266, 描述数量 = 586646, 类别 = 80
    # val2017 /: 图像数量 = 5000, 描述数量 = 25014
    # 处理后val2017: 图像数量 = 4952, 描述数量 = 24774, 类别 = 80
    # 最终数据集大小: 图像数量 = 122218, 描述数量 = 611420, 类别 = 80
    #
    # 进程已结束, 退出代码1