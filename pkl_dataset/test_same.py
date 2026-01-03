import pickle
import numpy as np

def array_equal_ignore_dtype(a, b):
    """忽略 dtype，仅比较 shape 和逐元素值"""
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        return False
    if a.shape != b.shape:
        return False
    return np.array_equal(a, b)


def compare_pkl_ignore_dtype(pkl1, pkl2):
    with open(pkl1, "rb") as f:
        data1 = pickle.load(f)
    with open(pkl2, "rb") as f:
        data2 = pickle.load(f)

    # 1. 类型检查
    if type(data1) != type(data2):
        print("❌ 顶层对象类型不同")
        return False

    # 2. dict 结构
    if isinstance(data1, dict):
        if data1.keys() != data2.keys():
            print("❌ 字典 key 不一致")
            print("data1 keys:", data1.keys())
            print("data2 keys:", data2.keys())
            return False

        for k in data1:
            v1, v2 = data1[k], data2[k]

            # --- numpy array ---
            if isinstance(v1, np.ndarray):
                if not array_equal_ignore_dtype(v1, v2):
                    print(f"❌ 键 {k} 的 numpy 数组不同")
                    print("shape:", v1.shape, v2.shape)
                    print("dtype:", v1.dtype, v2.dtype)
                    return False

            # --- list ---
            elif isinstance(v1, list):
                if len(v1) != len(v2):
                    print(f"❌ 键 {k} 的 list 长度不同: {len(v1)} vs {len(v2)}")
                    return False

                for i, (a, b) in enumerate(zip(v1, v2)):
                    if isinstance(a, np.ndarray):
                        if not array_equal_ignore_dtype(a, b):
                            print(f"❌ {k}[{i}] 的数组不同")
                            print("shape:", a.shape, b.shape)
                            print("dtype:", a.dtype, b.dtype)
                            return False
                    else:
                        if a != b:
                            print(f"❌ {k}[{i}] 不一致")
                            print("value1:", a)
                            print("value2:", b)
                            return False

            # --- 其他类型 ---
            else:
                if v1 != v2:
                    print(f"❌ 键 {k} 的值不同")
                    print("value1:", v1)
                    print("value2:", v2)
                    return False

    # 3. 非 dict 顶层对象
    else:
        if isinstance(data1, np.ndarray):
            if not array_equal_ignore_dtype(data1, data2):
                print("❌ 顶层 numpy 数组不同")
                return False
        else:
            if data1 != data2:
                print("❌ 顶层对象不一致")
                return False

    print("✅ 两个 pkl 在【忽略 dtype】的前提下完全一致")
    return True


# ================= 使用方式 =================
compare_pkl_ignore_dtype("coco2017_old.pkl", "coco2017.pkl")
