import torch
from torch.utils.data import Dataset
import os
import numpy as np

class LoadDataset_from_numpy1(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy1, self).__init__()

        # load files
        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            y_train = np.append(y_train, np.load(np_file)["y"])

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy, self).__init__()

        # load files

        X_train = np.load(np_dataset[0])["x"]
        #print(np_dataset)

        y_train = np.load(np_dataset[0])["y"]
        #print("y_train.shape", y_train.shape)

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            y_train = np.append(y_train, np.load(np_file)["y"], axis=0)
        #print("y_train.shape1", y_train.shape)
        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()



    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
class LoadDataset_from_numpy_UCI_train(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy_UCI_train, self).__init__()

        # load files

        X_train = np.load(np_dataset)["train_data"]
        #print(np_dataset)

        y_train = np.load(np_dataset)["train_labels"]
        #print("y_train.shape", y_train.shape)

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()



    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class LoadDataset_from_numpy_UCI(Dataset):
    # Initialize your data, download, etc.
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy_UCI, self).__init__()

        # load files

        X_train = np.load(np_dataset[0])["train_x"]
        # print(np_dataset)

        y_train = np.load(np_dataset[0])["train_y"]
        # print("y_train.shape", y_train.shape)

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["train_x"]))
            y_train = np.append(y_train, np.load(np_file)["train_y"], axis=0)
        # print("y_train.shape1", y_train.shape)
        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()



    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
class LoadDataset_from_numpy_UCI_test(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy_UCI_test, self).__init__()

        # load files
        X_train = np.load(np_dataset)["X_test"]
        # print(np_dataset)

        y_train = np.load(np_dataset)["test_labels"]

        # print("y_train.shape", y_train.shape)

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
class LoadDataset_from_numpy_xyz_hr(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy_xyz_hr, self).__init__()

        # load files

        X_train_x = np.load(np_dataset[0])["x"]
        X_train_hr = np.load(np_dataset[0])["x_hr"]
        #print(np_dataset)

        y_train = np.load(np_dataset[0])["y"]
        #print("y_train.shape", y_train.shape)

        for np_file in np_dataset[1:]:
            X_train_x = np.vstack((X_train_x, np.load(np_file)["x"]))
            X_train_hr = np.vstack((X_train_hr, np.load(np_file)["x_hr"]))
            y_train = np.append(y_train, np.load(np_file)["y"], axis=0)
        #print("y_train.shape1", y_train.shape)
        self.len = X_train_x.shape[0]
        self.x_data_x = torch.from_numpy(X_train_x)
        self.x_data_hr = torch.from_numpy(X_train_hr)
        self.y_data = torch.from_numpy(y_train).long()
        #print("self.y_data",self.y_data)



    def __getitem__(self, index):
        return self.x_data_x[index], self.x_data_hr[index], self.y_data[index]

    def __len__(self):
        return self.len
class ML_LoadDataset_from_numpy_xyz_hr(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(ML_LoadDataset_from_numpy_xyz_hr, self).__init__()

        # load files

        X_train_x = np.load(np_dataset[0])["x"]
        X_train_hr = np.load(np_dataset[0])["x_hr"]
        #print(np_dataset)

        y_train = np.load(np_dataset[0])["y"]
        y_ml_xyz_train = np.load(np_dataset[0])["ML_y_xyz"]
        y_ml_hr_train = np.load(np_dataset[0])["ML_y_hr"]
        #print("y_train.shape", y_train.shape)

        for np_file in np_dataset[1:]:
            X_train_x = np.vstack((X_train_x, np.load(np_file)["x"]))
            X_train_hr = np.vstack((X_train_hr, np.load(np_file)["x_hr"]))
            y_train = np.append(y_train, np.load(np_file)["y"], axis=0)
            y_ml_xyz_train = np.vstack((y_ml_xyz_train, np.load(np_file)["ML_y_xyz"]))
            y_ml_hr_train = np.vstack((y_ml_hr_train, np.load(np_file)["ML_y_hr"]))
        #print("y_train.shape1", y_train.shape)
        self.len = X_train_x.shape[0]
        self.x_data_x = torch.from_numpy(X_train_x)
        self.x_data_hr = torch.from_numpy(X_train_hr)
        self.y_data = torch.from_numpy(y_train).long()
        self.y_ml_xyz_train = torch.from_numpy(y_ml_xyz_train)
        self.y_ml_hr_train = torch.from_numpy(y_ml_hr_train)
        #print("self.y_data",self.y_data)



    def __getitem__(self, index):
        return self.x_data_x[index], self.x_data_hr[index], self.y_data[index], self.y_ml_xyz_train[index], self.y_ml_hr_train[index]

    def __len__(self):
        return self.len
class LoadDataset_from_numpy_xyz_hr_miss(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy_xyz_hr_miss, self).__init__()

        # load files

        X_train_x = np.load(np_dataset[0])["x"]
        X_train_hr = np.load(np_dataset[0])["x_hr"]
        X_train_x_miss = X_train_x
        X_train_hr_miss = X_train_hr
        #print(np_dataset)

        y_train = np.load(np_dataset[0])["y"]
        y_hr_miss_train = np.load(np_dataset[0])["hr_miss_labels"]
        y_xyz_miss_train = np.load(np_dataset[0])["xyz_miss_labels"]
        y_hr_modal_train = np.load(np_dataset[0])["hr_modal_labels"]
        y_xyz_modal_train = np.load(np_dataset[0])["xyz_modal_labels"]
        #print("y_train.shape", y_train.shape)

        # 生成示例数据
        bh, th, ch, dh = X_train_hr_miss.shape
        bx, tx, cx, dx = X_train_x_miss.shape

        X_train_hr_miss = X_train_hr_miss.reshape(-1, ch, dh)
        X_train_x_miss = X_train_x_miss.reshape(-1, cx, dx)
        y_hr_miss_train = y_hr_miss_train.reshape(-1)
        y_xyz_miss_train = y_xyz_miss_train.reshape(-1)

        ###采样出同一个模态下的不缺失数据
        #HR
        # 找到标签为 0 的数据的索引位置
        zero_indices = np.where(y_hr_miss_train == 0)[0]
        # 从所有标签为 1 的数据中随机选择与标签为 0 的数据数量相同的数据的索引位置
        one_indices = np.where(y_hr_miss_train == 1)[0]
        np.random.seed(40)#42
        random_one_indices = np.random.choice(one_indices, size=len(zero_indices), replace=True)
        # 将标签为 0 的数据的值分别替换为标签为 1 的随机选择的数据的值
        X_train_hr_miss[zero_indices] = X_train_hr_miss[random_one_indices]
        X_train_hr_miss = X_train_hr_miss.reshape(bh, th, ch, dh)
        y_hr_miss_train = y_hr_miss_train.reshape(-1, 20)

        # XYZ
        # 找到标签为 0 的数据的索引位置
        zero_indices = np.where(y_xyz_miss_train == 0)[0]
        # 从所有标签为 1 的数据中随机选择与标签为 0 的数据数量相同的数据的索引位置
        one_indices = np.where(y_xyz_miss_train == 1)[0]
        np.random.seed(42)
        random_one_indices = np.random.choice(one_indices, size=len(zero_indices), replace=True)
        # 将标签为 0 的数据的值分别替换为标签为 1 的随机选择的数据的值
        X_train_x_miss[zero_indices] = X_train_x_miss[random_one_indices]
        X_train_x_miss = X_train_x_miss.reshape(bx, tx, cx, dx)
        y_xyz_miss_train = y_xyz_miss_train.reshape(-1, 20)


        for np_file in np_dataset[1:]:
            y_hr_miss_train_post = np.load(np_file)["hr_miss_labels"]
            y_xyz_miss_train_post = np.load(np_file)["xyz_miss_labels"]
            y_hr_miss_train_post = y_hr_miss_train_post.reshape(-1)
            y_xyz_miss_train_post = y_xyz_miss_train_post.reshape(-1)
            X_train_hr_miss_post = np.load(np_file)["x_hr"]
            X_train_x_miss_post = np.load(np_file)["x"]
            bh, th, ch, dh = X_train_hr_miss_post.shape
            bx, tx, cx, dx = X_train_x_miss_post.shape
            X_train_hr_miss_post = X_train_hr_miss_post.reshape(-1, ch, dh)
            X_train_x_miss_post = X_train_x_miss_post.reshape(-1, cx, dx)
            ###采样出同一个模态下的不缺失数据
            # HR
            # 找到标签为 0 的数据的索引位置
            zero_indices = np.where(y_hr_miss_train_post == 0)[0]
            # 从所有标签为 1 的数据中随机选择与标签为 0 的数据数量相同的数据的索引位置
            one_indices = np.where(y_hr_miss_train_post == 1)[0]
            np.random.seed(42)
            random_one_indices = np.random.choice(one_indices, size=len(zero_indices), replace=True)
            # 将标签为 0 的数据的值分别替换为标签为 1 的随机选择的数据的值
            X_train_hr_miss_post[zero_indices] = X_train_hr_miss_post[random_one_indices]
            X_train_hr_miss_post = X_train_hr_miss_post.reshape(bh, th, ch, dh)
            #y_hr_miss_train_post = y_hr_miss_train_post.reshape(-1, 20)

            # XYZ
            # 找到标签为 0 的数据的索引位置
            zero_indices = np.where(y_xyz_miss_train_post == 0)[0]
            # 从所有标签为 1 的数据中随机选择与标签为 0 的数据数量相同的数据的索引位置
            one_indices = np.where(y_xyz_miss_train_post == 1)[0]
            np.random.seed(42)
            random_one_indices = np.random.choice(one_indices, size=len(zero_indices), replace=True)
            # 将标签为 0 的数据的值分别替换为标签为 1 的随机选择的数据的值
            X_train_x_miss_post[zero_indices] = X_train_x_miss_post[random_one_indices]
            X_train_x_miss_post = X_train_x_miss_post.reshape(bx, tx, cx, dx)
            #y_xyz_miss_train_post = y_xyz_miss_train_post.reshape(-1, 20)


            X_train_x = np.vstack((X_train_x, np.load(np_file)["x"]))
            X_train_hr = np.vstack((X_train_hr, np.load(np_file)["x_hr"]))
            X_train_hr_miss = np.vstack((X_train_hr_miss, X_train_hr_miss_post))
            X_train_x_miss = np.vstack((X_train_x_miss, X_train_x_miss_post))
            y_train = np.append(y_train, np.load(np_file)["y"], axis=0)
            y_hr_miss_train = np.append(y_hr_miss_train, np.load(np_file)["hr_miss_labels"], axis=0)
            y_xyz_miss_train = np.append(y_xyz_miss_train, np.load(np_file)["xyz_miss_labels"], axis=0)
            y_hr_modal_train = np.append(y_hr_modal_train, np.load(np_file)["hr_modal_labels"], axis=0)
            y_xyz_modal_train = np.append(y_xyz_modal_train, np.load(np_file)["xyz_modal_labels"], axis=0)

            #y_hr_miss_train = np.vstack((y_hr_miss_train, np.load(np_file)["hr_miss_labels"]))
            #y_xyz_miss_train = np.vstack((y_xyz_miss_train, np.load(np_file)["xyz_miss_labels"]))
            #y_hr_modal_train = np.vstack((y_hr_modal_train, np.load(np_file)["hr_modal_labels"]))
            #y_xyz_modal_train = np.vstack((y_xyz_modal_train, np.load(np_file)["xyz_modal_labels"]))
        #print("y_train.shape1", y_train.shape)





        self.len = X_train_x.shape[0]
        self.x_data_x = torch.from_numpy(X_train_x)
        self.x_data_hr = torch.from_numpy(X_train_hr)
        self.X_train_x_miss = torch.from_numpy(X_train_x_miss)
        self.X_train_hr_miss = torch.from_numpy(X_train_hr_miss)

        self.y_data = torch.from_numpy(y_train).long()
        self.y_hr_miss = torch.from_numpy(y_hr_miss_train).long()
        self.y_xyz_miss= torch.from_numpy(y_xyz_miss_train).long()
        self.y_hr_modal = torch.from_numpy(y_hr_modal_train).long()
        self.y_xyz_modal = torch.from_numpy(y_xyz_modal_train).long()
        """
        print("####")
        print(self.x_data_x.shape)
        print(self.x_data_hr.shape)
        print(self.X_train_x_miss.shape)
        print(self.X_train_hr_miss.shape)
        print(self.y_data.shape)
        print(self.y_hr_miss.shape)
        print(self.y_xyz_miss.shape)
        print(self.y_hr_modal.shape)
        print(self.y_xyz_modal.shape)
        """

        #print("self.y_data",self.y_data.shape)



    def __getitem__(self, index):
        #print(index)
        #print(self.y_data[index].shape)
        return self.x_data_x[index], self.x_data_hr[index], self.X_train_x_miss[index], self.X_train_hr_miss[index], self.y_data[index], self.y_xyz_miss[index], self.y_hr_miss[index], self.y_xyz_modal[index], self.y_hr_modal[index]

    def __len__(self):
        return self.len


class LoadDataset_from_numpy_edf(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy_edf, self).__init__()

        # load files

        X_train_EEG = np.load(np_dataset[0])["x_EEG"]
        X_train_EOG = np.load(np_dataset[0])["x_EOG"]
        X_train_EEG_miss = np.load(np_dataset[0])["x_EEG_miss"]
        X_train_EOG_miss = np.load(np_dataset[0])["x_EOG_miss"]
        #print(np_dataset)
        y_train = np.load(np_dataset[0])["y"]
        y_train_EEG_miss = np.load(np_dataset[0])["EEG_miss_labels"]
        y_train_EOG_miss = np.load(np_dataset[0])["EOG_miss_labels"]
        y_train_EEG_modal = np.load(np_dataset[0])["EEG_modal_labels"]
        y_train_EOG_modal = np.load(np_dataset[0])["EOG_modal_labels"]

        #print("y_train.shape", y_train.shape)

        for np_file in np_dataset[1:]:
            X_train_EEG = np.vstack((X_train_EEG, np.load(np_file)["x_EEG"]))
            X_train_EOG = np.vstack((X_train_EOG, np.load(np_file)["x_EOG"]))
            X_train_EEG_miss = np.vstack((X_train_EEG_miss, np.load(np_file)["x_EEG_miss"]))
            X_train_EOG_miss = np.vstack((X_train_EOG_miss, np.load(np_file)["x_EOG_miss"]))
            y_train = np.append(y_train, np.load(np_file)["y"], axis=0)
            y_train_EEG_miss = np.append(y_train_EEG_miss, np.load(np_file)["EEG_miss_labels"], axis=0)
            y_train_EOG_miss = np.append(y_train_EOG_miss, np.load(np_file)["EOG_miss_labels"], axis=0)
            y_train_EEG_modal = np.append(y_train_EEG_modal, np.load(np_file)["EEG_modal_labels"], axis=0)
            y_train_EOG_modal = np.append(y_train_EOG_modal, np.load(np_file)["EOG_modal_labels"], axis=0)
        #print("y_train.shape1", y_train.shape)
        self.len = X_train_EEG.shape[0]
        self.X_train_EEG = torch.from_numpy(X_train_EEG)
        self.X_train_EOG = torch.from_numpy(X_train_EOG)
        self.X_train_EEG_miss = torch.from_numpy(X_train_EEG_miss)
        self.X_train_EOG_miss = torch.from_numpy(X_train_EOG_miss)
        self.y_train = torch.from_numpy(y_train).long()
        self.y_train_EEG_miss = torch.from_numpy(y_train_EEG_miss).long()
        self.y_train_EOG_miss = torch.from_numpy(y_train_EOG_miss).long()
        self.y_train_EEG_modal = torch.from_numpy(y_train_EEG_modal).long()
        self.y_train_EOG_modal = torch.from_numpy(y_train_EOG_modal).long()
        #print("self.y_data",self.y_data)



    def __getitem__(self, index):
        return self.X_train_EEG[index], self.X_train_EOG[index], self.X_train_EEG_miss[index], self.X_train_EOG_miss[index], self.y_train[index], self.y_train_EEG_miss[index],\
               self.y_train_EOG_miss[index], self.y_train_EEG_modal[index], self.y_train_EOG_modal[index]

    def __len__(self):
        return self.len
class LoadDataset_from_numpy_WESAD_miss(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy_WESAD_miss, self).__init__()

        X_train_ECG = np.load(np_dataset[0])["x_ECG"]
        X_train_EDA = np.load(np_dataset[0])["x_EDA"]
        X_train_ECG_miss = np.load(np_dataset[0])["x_ECG_miss"]
        X_train_EDA_miss = np.load(np_dataset[0])["x_EDA_miss"]
        #print(np_dataset)
        y_train = np.load(np_dataset[0])["y"]
        y_train_ECG_miss = np.load(np_dataset[0])["y_ECG_miss"]
        y_train_EDA_miss = np.load(np_dataset[0])["y_EDA_miss"]
        y_train_ECG_modal = np.load(np_dataset[0])["ECG_modal_labels"]
        y_train_EDA_modal = np.load(np_dataset[0])["EDA_modal_labels"]

        #print("y_train.shape", y_train.shape)

        for np_file in np_dataset[1:]:
            X_train_ECG = np.vstack((X_train_ECG, np.load(np_file)["x_ECG"]))
            X_train_EDA = np.vstack((X_train_EDA, np.load(np_file)["x_EDA"]))
            X_train_ECG_miss = np.vstack((X_train_ECG_miss, np.load(np_file)["x_ECG_miss"]))
            X_train_EDA_miss = np.vstack((X_train_EDA_miss, np.load(np_file)["x_EDA_miss"]))
            y_train = np.append(y_train, np.load(np_file)["y"], axis=0)
            y_train_ECG_miss = np.append(y_train_ECG_miss, np.load(np_file)["y_ECG_miss"], axis=0)
            y_train_EDA_miss = np.append(y_train_EDA_miss, np.load(np_file)["y_EDA_miss"], axis=0)
            y_train_ECG_modal = np.append(y_train_ECG_modal, np.load(np_file)["ECG_modal_labels"], axis=0)
            y_train_EDA_modal = np.append(y_train_EDA_modal, np.load(np_file)["EDA_modal_labels"], axis=0)
        #print("y_train.shape1", y_train.shape)
        self.len = X_train_ECG.shape[0]
        self.X_train_ECG = torch.from_numpy(X_train_ECG)
        self.X_train_EDA = torch.from_numpy(X_train_EDA)
        self.X_train_ECG_miss = torch.from_numpy(X_train_ECG_miss)
        self.X_train_EDA_miss = torch.from_numpy(X_train_EDA_miss)
        self.y_train = torch.from_numpy(y_train).long()
        self.y_train_ECG_miss = torch.from_numpy(y_train_ECG_miss).long()
        self.y_train_EDA_miss = torch.from_numpy(y_train_EDA_miss).long()
        self.y_train_ECG_modal = torch.from_numpy(y_train_ECG_modal).long()
        self.y_train_EDA_modal = torch.from_numpy(y_train_EDA_modal).long()
        #print("self.y_data",self.y_data)



    def __getitem__(self, index):
        return self.X_train_ECG[index], self.X_train_EDA[index], self.X_train_ECG_miss[index], self.X_train_EDA_miss[index], self.y_train[index], self.y_train_ECG_miss[index],\
               self.y_train_EDA_miss[index], self.y_train_ECG_modal[index], self.y_train_EDA_modal[index]

    def __len__(self):
        return self.len






def data_generator_np_miss_edf(training_files, subject_files, testing_files, batch_size):
    train_dataset = LoadDataset_from_numpy_edf(training_files)

    val_dataset = LoadDataset_from_numpy_edf(subject_files)

    test_dataset = LoadDataset_from_numpy_edf(testing_files)


    # to calculate the ratio for the CAL
    all_ys = np.concatenate((train_dataset.y_train, val_dataset.y_train))
    all_ys = np.concatenate((all_ys, test_dataset.y_train))
    all_ys_f = all_ys.flatten()
    #print("a",all_ys.shape)
    train_ys = np.array(train_dataset.y_train)
    train_ys_f = train_ys.flatten()
    #print(" train_ys",train_ys.shape)
    train_ys_f = train_ys_f.tolist()
    all_ys_f = all_ys_f.tolist()
    #print(np.unique(all_ys))
    num_classes = len(np.unique(all_ys_f))
    #print("a",num_classes)
    counts = [all_ys_f.count(i) for i in range(num_classes)]
    counts_train = [ train_ys_f.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, val_loader, test_loader, counts, counts_train
