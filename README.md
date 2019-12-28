```bash
#coding=utf-8
import logging
import csv
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

logging.getLogger().setLevel(logging.INFO)

class Model:
    def __init__(self):
        self.count = 0
        self.Dicts = {}

    def preProcess(self, path):
        # 计算最小自由能
        mdata = open('../Script/Mapping/Mfe/NavMfe.txt', 'r').readlines()
        count = 0
        mfe = []
        logging.info("START CALCULATE MFE")
        for key in mdata:
            count = count + 1
            if count % 3 == 2:
                s = key.strip()
            if count % 3 == 0:
                s = s + "," + key.split(" ")[-1].split("-")[-1].split(")")[0]
                mfe.append(s)
        logging.info("END CALCULATE MFE")

        # 计算Motif
        logging.info("START Pre MOTIF")
        level1 = ["A", "G", "C", "T"]
        level2 = []
        level3 = []
        for i in level1:
            for j in level1:
                level2.append(i + j)
        for i in level1:
            for j in level2:
                level3.append(i + j)
        logging.info("END Pre MOTIF")

        # 转码
        encode = {
            "N" : [0, 0, 0, 0],
            "A" : [0, 0, 0, 1],
            "G" : [0, 0, 1, 0],
            "C" : [0, 1, 0, 0],
            "U" : [1, 0, 0, 0]
        }
        headers = ["sequence", "length", "GC", "mfe",
                    "l1_1","l1_2","l1_3","l1_4","l2_1","l2_2","l2_3","l2_4","l3_1","l3_2","l3_3","l3_4","l4_1","l4_2","l4_3","l4_4",
                    "l5_1","l5_2","l5_3","l5_4","l6_1","l6_2","l6_3","l6_4","l7_1","l7_2","l7_3","l7_4","l8_1","l8_2","l8_3","l8_4",
                    "l9_1","l9_2","l9_3","l9_4","l10_1","l10_2","l10_3","l10_4","l11_1","l11_2","l11_3","l11_4","l12_1","l12_2","l12_3","l12_4",
                    "l13_1","l13_2","l13_3","l13_4","l14_1","l14_2","l14_3","l14_4","l15_1","l15_2","l15_3","l15_4","l16_1","l16_2","l16_3","l16_4",
                    "l17_1","l17_2","l17_3","l17_4","l18_1","l18_2","l18_3","l18_4","l19_1","l19_2","l19_3","l19_4","l20_1","l20_2","l20_3","l20_4",
                    "l21_1","l21_2","l21_3","l21_4","l22_1","l22_2","l22_3","l22_4","l23_1","l23_2","l23_3","l23_4","l24_1","l24_2","l24_3","l24_4",
                    "l25_1","l25_2","l25_3","l25_4",
                    "3UTR_1","3UTR_2","3UTR_3","3UTR_4","3UTR_5","3UTR_6","3UTR_7","3UTR_8",
                    "5UTR_1","5UTR_2","5UTR_3","5UTR_4","5UTR_5","5UTR_6","5UTR_7","5UTR_8",
                    "m1A1", "m1A2", "m1A3", "m1A4",
                    "m2A1", "m2A2", "m2A3", "m2A4", "m2A5", "m2A6", "m2A7", "m2A8", "m2A9", "m2A10", "m2A11", "m2A12",
                    "m2A13", "m2A14", "m2A15", "m2A16",
                    "m3A1", "m3A2", "m3A3", "m3A4", "m3A5", "m3A6", "m3A7", "m3A8", "m3A9", "m3A10", "m3A11", "m3A12", "m3A13",
                    "m3A14", "m3A15", "m3A16","m3A17", "m3A18", "m3A19", "m3A20", "m3A21", "m3A22", "m3A23", "m3A24", "m3A25",
                    "m3A26", "m3A27", "m3A28", "m3A29", "m3A30", "m3A31", "m3A32", "m3A33", "m3A34", "m3A35", "m3A36", "m3A37",
                    "m3A38", "m3A39", "m3A40", "m3A41", "m3A42", "m3A43", "m3A44", "m3A45", "m3A46", "m3A47", "m3A48", "m3A49",
                    "m3A50", "m3A51", "m3A52", "m3A53", "m3A54", "m3A55", "m3A56", "m3A57", "m3A58", "m3A59", "m3A60", "m3A61",
                    "m3A62", "m3A63", "m3A64",
                    "class"
                    ]
        data = open(path, 'r').readlines()
        ls = []

        logging.info("START CALCULATE ALLContent")
        for key in data:
            tcount = 0
            temp = []
            for i in key.strip():
                if i == 'G' or i == 'C':
                    tcount = tcount + 1
            mt = ""
            for m in mfe:
                if key.strip() == m.split(",")[0]:
                    mt = round(float(m.split(",")[1]))
            temp = [key.strip(), len(key.strip()), round(tcount/len(key.strip())*10), mt]

            addrls = []
            for ik in range(len(key.strip())):
                addrls.extend(encode[key[ik]])
            for ik in range(len(key.strip()), 25):
                addrls.extend(encode["N"])
            addrls.extend(encode[key[0]])
            addrls.extend(encode[key[1]])
            addrls.extend(encode[key[len(key.strip())-2]])
            addrls.extend(encode[key[len(key.strip())-1]])
            temp.extend(addrls)
            # level1
            for l1 in level1:
                acount = 0
                t1 = ""
                for i1 in key.strip():
                    t1 = t1 + i1
                    if i1 == l1:
                        acount = acount + 1
                temp.append(acount)
            # level2
            for l2 in level2:
                acount = 0
                for i2 in range(0, len(key.strip()) - 3):
                    if l2 == key[i2:i2 + 2]:
                        acount = acount + 1
                temp.append(acount)
            # level3
            for l3 in level3:
                acount = 0
                for i3 in range(0, len(key.strip()) - 4):
                    if l3 == key[i3:i3 + 3]:
                        acount = acount + 1
                temp.append(acount)
            temp.append(1)
            ls.append(temp)
        logging.info("END CALCULATE ALLContent")

        logging.info("START Writting")
        with open("../Script/Mapping/Mfe/NavSample.csv", "w") as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(ls)
        logging.info("END Writting")

    @staticmethod
    def DataInit():
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from random import shuffle

        df = pd.read_csv("../Script/Mapping/Mfe/Sample30.csv")
        cols = list(df.columns.values)
        cols_data = copy.deepcopy(cols)
        cols_data.remove('class')
        x_data = df[list(cols_data)]
        x = np.array(x_data)
        x = StandardScaler().fit_transform(x)
        y_data = df['class']
        y = np.array(y_data)
        data = np.insert(x, x[0].size, values=y, axis=1)
        np.random.shuffle(data)
        y = data[:,data[0].size-1]
        x = np.delete(data, -1, axis=1)
        
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=1)
        return X_train, X_test, Y_train, Y_test

    def KnnDemo(self):
        X_train, X_test, Y_train, Y_test = Model.DataInit()
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import precision_score
        
        for i in range(10):
            knn = KNeighborsClassifier(n_neighbors=25)
            knn.fit(X_train, Y_train)
            y_true, y_pred = Y_test, knn.predict(X_test)
            print(precision_score(y_true, y_pred))


        print(classification_report(y_true, y_pred))
        # print(grid.best_estimator_)
        y_pred_pro = knn.predict_proba(X_test)
        y_scores = pd.DataFrame(y_pred_pro, columns=knn.classes_.tolist())[1].values
        auc_value = roc_auc_score(y_true, y_scores)
        print("roc:",auc_value)

        # 训练集
        y_true_last, y_pred_last = Y_train, knn.predict(X_train)
        p_pred_last = knn.predict_proba(X_train)
        y_scores_last = pd.DataFrame(p_pred_last, columns=knn.classes_.tolist())[1].values
        acu_train = roc_auc_score(y_true_last, y_scores_last)
        print("roc_train:", acu_train)
        # clf = svm.SVC(kernel='rbf', class_weight='balanced')
        # clf.fit(X_train, y_train)
        # 绘制ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='blue', label='AUC = %0.4f' % auc_value)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def SVMDemo(self):
        X_train, X_test, Y_train, Y_test = Model.DataInit()
        from sklearn import svm
        from sklearn.model_selection import GridSearchCV

        param_test1 = {
            'C': [x for x in np.arange(0.5, 2, 0.1)],
            'gamma': [x for x in np.arange(0.01, 0.1, 0.01)],
            'tol': [x for x in np.arange(0.01, 0.1, 0.01)],
        }
        scores = ['precision', 'recall']
        print("# Tuning hyper-parameters for roc_auc")
        print()
        grid = GridSearchCV(estimator=svm.SVC(probability=True),
                            param_grid=param_test1, scoring='accuracy', refit=True, cv=5)
        # accuracy,roc_auc
        grid.fit(X_train, Y_train)
        print(grid.best_params_)
        print(grid.best_score_)
        print("Grid scores on development set:")
        print()
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        from sklearn.metrics import classification_report
        y_true, y_pred = Y_test, grid.predict(X_test)

        # 打印在测试集上的预测结果与真实值的分数
        print(classification_report(y_true, y_pred))
        print("================================")
        print(classification_report(Y_train, grid.predict(X_train)))
        y_pred_pro = grid.predict_proba(X_test)
        y_scores = pd.DataFrame(y_pred_pro, columns=grid.classes_.tolist())[1].values
        auc_value = roc_auc_score(y_true, y_scores)
        print(auc_value)

        # 训练集
        y_pred_last = grid.predict_proba(X_train)
        y_scores_last = pd.DataFrame(y_pred_last, columns=grid.classes_.tolist())[1].values
        auc_value_last = roc_auc_score(Y_train, y_scores_last)
        print(auc_value_last)

        # 绘制ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='blue', label='AUC = %0.4f' % auc_value)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def RandomSvmModel(self):
        sample = "../Script/Mapping/Mfe/Sample.csv"
        df = pd.read_csv(sample)  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
        cols = list(df.columns.values)
        cols_data = copy.deepcopy(cols)
        #cols_data.remove('sequence')
        cols_data.remove('class')
        x_data = df[list(cols_data)]  # 抽取作为训练数据的各属性值
        x = np.array(x_data)
        print(len(x))
        y_data = df['class']  # 最后一列作为每行对应的标签label
        y = np.array(y_data)
        print(y)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
        ############################ -*- coding: 得到特征重要性排序 -*-######################
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        param_test1 = {'n_estimators': range(100, 1000, 100)}
        gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                                 min_samples_leaf=20, max_depth=8, max_features='sqrt',
                                                                 random_state=10),
                                param_grid=param_test1, scoring='roc_auc', cv=5)
        gsearch1.fit(X_train, y_train)

        param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(2, 200, 10)}
        gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=350, min_samples_split=10,
                                                                 min_samples_leaf=20, max_depth=8, max_features='sqrt',
                                                                 random_state=10),
                                param_grid=param_test2, scoring='roc_auc', cv=5)
        gsearch2.fit(X_train, y_train)

        # 再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
        param_test3 = {'min_samples_split': range(10, 150, 20), 'min_samples_leaf': range(10, 60, 10)}
        gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=350, min_samples_split=42,
                                                                 min_samples_leaf=20, max_depth=9, max_features='sqrt',
                                                                 random_state=10),
                                param_grid=param_test3, scoring='roc_auc', cv=5)
        gsearch3.fit(X_train, y_train)

        param_test4 = {'max_features': range(3, 50, 2)}
        gsearch4 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=350, min_samples_split=10,
                                                                 min_samples_leaf=10, max_depth=9, random_state=10),
                                param_grid=param_test4, scoring='roc_auc', cv=5)
        gsearch4.fit(X_train, y_train)
        gsearch4.best_params_
        feat_lables = cols

        forest = RandomForestClassifier(n_estimators=350, min_samples_split=10,
                                        min_samples_leaf=10, max_depth=9, random_state=10, max_features=50,
                                        oob_score=True)
        forest.fit(X_train, y_train)
        importance = forest.feature_importances_
        imp_result = np.argsort(importance)[::-1]
        feature = []
        x_d = []
        y_d = []
        for i in range(X_train.shape[1]):
            temp = imp_result[i]
            #print("%2d. %-*s %f" % (i, 30, cols_data[temp], importance[temp]))
            x_d.append(cols_data[temp])
            y_d.append(importance[temp])
            feature.append(cols_data[temp])
        print(feature[0: 30])
        import matplotlib.pyplot as plt
        plt.bar(x_d[0:30], height=y_d[0:30], tick_label=x_d[0:30])
        plt.setp(plt.gca().get_xticklabels(), rotation=80, horizontalalignment='right')
        plt.show()

        sample = "../Script/Mapping/Mfe/Sample.csv"
        df = pd.read_csv(sample)  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
        cols = list(df.columns.values)
        cols_data = copy.deepcopy(cols)
        #cols_data.remove('sequence')
        cols_data.remove('class')
        x_data = df[list(feature[0: 49])]  # 抽取作为训练数据的各属性值
        x = np.array(x_data)
        y_data = df['class']  # 最后一列作为每行对应的标签label
        y = np.array(y_data)
        import matplotlib.pyplot as plt
        plt.bar(range(len(importance[0:30])), importance[0:30], tick_label=cols_data[0:30])
        plt.setp(plt.gca().get_xticklabels(), rotation=80, horizontalalignment='right')
        plt.show()

        from sklearn import svm
        from sklearn.model_selection import GridSearchCV
        import matplotlib.pyplot as plt

        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import classification_report


        param_test1 = {
            'C': [x for x in np.arange(0.5, 2, 0.1)],
            'gamma': [x for x in np.arange(0.01, 0.1, 0.01)],
            'tol': [x for x in np.arange(0.01, 0.1, 0.01)],
        }

        scores = ['precision', 'recall']

        print("# Tuning hyper-parameters for roc_auc")
        print()
        grid = GridSearchCV(estimator=svm.SVC(probability=True),
                            param_grid=param_test1, scoring='accuracy', refit=True, cv=5)
        # accuracy,roc_auc
        grid.fit(X_train, y_train)
        print(grid.best_params_)
        print(grid.best_score_)
        print("Grid scores on development set:")
        print()
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, grid.predict(X_test)
        # 打印在测试集上的预测结果与真实值的分数
        print(classification_report(y_true, y_pred))
        print("================================")
        print(classification_report(y_train, grid.predict(X_train)))

        print(grid.best_estimator_)
        y_pred_pro = grid.predict_proba(X_test)
        y_scores = pd.DataFrame(y_pred_pro, columns=grid.classes_.tolist())[1].values
        auc_value = roc_auc_score(y_true, y_scores)
        # clf = svm.SVC(kernel='rbf', class_weight='balanced')
        # clf.fit(X_train, y_train)
        # 绘制ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='blue', label='AUC = %0.4f' % auc_value)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def LogisticRegression(self):
        X_train, X_test, Y_train, Y_test = Model.DataInit()
        from sklearn import metrics
        from sklearn.metrics import precision_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        # print(model)  # 输出模型
        # make predictions
        expected = Y_test  # 测试样本的期望输出
        predicted = model.predict(X_test)  # 测试样本预测
        return accuracy_score(expected, predicted), precision_score(expected, predicted), recall_score(expected, predicted), f1_score(expected, predicted)
        
        # 输出结果
        print(metrics.classification_report(expected, predicted))

        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        y_true, y_pred = Y_test, model.predict(X_test)
        y_pred_pro = model.predict_proba(X_test)
        y_scores = pd.DataFrame(y_pred_pro, columns=model.classes_.tolist())[1].values
        auc_value = roc_auc_score(y_true, y_scores)
        print(auc_value)

        # 训练集
        y_true_last, y_pred_last = Y_train, model.predict(X_train)
        p_pred_last = model.predict_proba(X_train)
        y_scores_last = pd.DataFrame(p_pred_last, columns=model.classes_.tolist())[1].values
        acu_train = roc_auc_score(y_true_last, y_scores_last)
        print("roc_train:", acu_train)

        # 绘制ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)
        plt.figure(figsize=(5,5))
        lw = 2
        plt.plot(fpr, tpr, color='blue', label='AUC = %0.4f' % auc_value)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def MLPModel(self, x):
        import tensorflow as tf
        w1 = tf.Variable(tf.random_normal([1, 100], stddev=1, seed=1))
        w2 = tf.Variable(tf.random_normal([100, 50], stddev=1, seed=1))
        w3 = tf.Variable(tf.random_normal([50, 100], stddev=1, seed=1))
        w4 = tf.Variable(tf.random_normal([100, 1], stddev=1, seed=1))
        y1 = tf.matmul(x, w1)
        y11 = tf.nn.relu(y1)
        y2 = tf.matmul(y11, w2)
        y22 = tf.nn.relu(y2)
        y3 = tf.matmul(y22, w3)
        y33 = tf.nn.relu(y3)
        y = tf.matmul(y33, w4)
        # y = tf.nn.softmax(y4)
        return y

    def MLPtrain(self, x0, y0):
        import tensorflow as tf
        x = tf.placeholder(tf.float32, shape=(None, 203), name='x-input')
        y = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

        y_ = self.MLPModel(x)
        loss = tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                              +(1-y_) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            steps = 1000
            for i in range(steps):
                sess.run(train_step, feed_dict={x:x0, y_:y0})


if __name__ == '__main__':
    modle = Model()
    # sample = "../Script/Mapping/Mfe/Sample.csv"
    # potus = list(csv.reader(open(sample)))
    #
    # x = []
    # y = []
    # for i in range(1, len(potus)):
    #     #temp = [int(x) for x in potus[i][0:len(potus[i])-1]]
    #     x.append([int(x) for x in potus[i][0:len(potus[i])-1]])
    #     y.append(int(potus[i][len(potus[i])-1]))
    # print(len(x[0]))

    # 特征提取
    # modle.preProcess('../Script/Mapping/NavSample_seq.fasta')
    # KNN
    # modle.KnnDemo()
    # SVM
    # modle.SVMDemo()
    # RandowASVM
    # modle.RandomSvmModel()
    headers = ["a", "p", "r", "f"]
    contents = []
    for i in range(100):
        content = []
        a, p, r, f = modle.LogisticRegression()
        contents.append([a,p,r,f])
    # 写文件
    with open("../ML/LRparm.csv", "w") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(contents)
    # modle.DataInit()

```
