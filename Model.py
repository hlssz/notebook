``` bash
#coding=utf-8
import logging
import csv
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from ML.feature import feature

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
        headers = feature
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
    def DataInit(evalute=False):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        if evalute:
            dftrain = pd.read_csv("../Script/Mapping/Mfe/SamTrain.csv")
            cols = list(dftrain.columns.values)
            cols_data = copy.deepcopy(cols)
            cols_data.remove('class')
            x_data = dftrain[list(cols_data)]
            x = np.array(x_data)
            X_train = StandardScaler().fit_transform(x)
            y_data = dftrain['class']
            Y_train = np.array(y_data)

            dftest = pd.read_csv("../Script/Mapping/Mfe/SamTest.csv")
            cols = list(dftest.columns.values)
            cols_data = copy.deepcopy(cols)
            cols_data.remove('class')
            x_data = dftest[list(cols_data)]
            x = np.array(x_data)
            X_test = StandardScaler().fit_transform(x)
            y_data = dftest['class']
            Y_test = np.array(y_data)
        else:
            df = pd.read_csv("../Script/Mapping/Mfe/SamTrain.csv")
            cols = list(df.columns.values)
            cols_data = copy.deepcopy(cols)
            cols_data.remove('class')
            x_data = df[list(cols_data)]
            x = np.array(x_data)
            x = StandardScaler().fit_transform(x)
            y_data = df['class']
            y = np.array(y_data)
            data = np.insert(x, x[0].size, values=y, axis=1)
            # np.random.shuffle(data)
            y = data[:, data[0].size - 1]
            x = np.delete(data, -1, axis=1)
            X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=1)

        # 先保存测试集和数据集
        # headers = feature
        # train_content = []
        # temp = []
        # Y_train = Y_train.astype(np.float64)
        # for i in range(len(X_train)):
        #     temp = []
        #     for j in X_train[i]:
        #         temp.append(j.item())
        #     temp.append(int(Y_train[i]))
        #     train_content.append(temp)
        # print(len(train_content))
        # test_content = []
        # Y_test = Y_test.astype(np.float64)
        # for i in range(len(X_test)):
        #     temp = []
        #     for j in X_test[i]:
        #         temp.append(j.item())
        #     temp.append(Y_test[i])
        #     test_content.append(temp)
        # print(len(test_content))
        # with open("../ML/Train.csv", "w") as ftrain:
        #     f_csv = csv.writer(ftrain)
        #     f_csv.writerow(headers)
        #     f_csv.writerows(train_content)
        # with open("../ML/Test.csv", "w") as ftest:
        #     f_csv = csv.writer(ftest)
        #     f_csv.writerow(headers)
        #     f_csv.writerows(test_content)
        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def FigAuc(y_true, y_scores, auc_value):
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

    def KnnDemo(self):
        X_train, X_test, Y_train, Y_test = Model.DataInit()
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score


        k_scores = [0 for i in range(863)]
        k_score = [0 for i in range(863)]
        for i in range(100):
            k_score = []
            for k in range(1, 863):
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
                k_score.append(scores.mean())
            k_scores = [a+b for a,b in zip(k_scores, k_score)]
        for i in range(len(k_scores)):
            k_scores[i] = round(k_scores[i], 4)
        print(k_scores)
        plt.plot(range(1, 863), k_scores)
        plt.xlabel("Value Of K")
        plt.ylabel("Cross-Validated Accuracy")
        plt.show()
        return

        knn = KNeighborsClassifier(n_neighbors=198)
        knn.fit(X_train, Y_train)
        probility = knn.predict_proba(X_test)
        score = knn.score(X_test, Y_test, sample_weight=None)

        # 测试集
        y_true, y_pred = Y_test, knn.predict(X_test)
        # 打印在测试集上的预测结果与真实值的分数
        print(classification_report(y_true, y_pred))
        # print(grid.best_estimator_)
        y_pred_pro = knn.predict_proba(X_test)
        y_scores = pd.DataFrame(y_pred_pro, columns=knn.classes_.tolist())[1].values
        auc_value = roc_auc_score(y_true, y_scores)
        print("roc:",auc_value)
        expected = y_true
        predicted = y_pred
        #return accuracy_score(expected, predicted), precision_score(expected, predicted), recall_score(expected, predicted), f1_score(expected, predicted), auc_value

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
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        # print(model)  # 输出模型
        # make predictions
        expected = Y_test  # 测试样本的期望输出
        predicted = model.predict(X_test)  # 测试样本预测
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
        expected = y_true
        predicted = y_pred
        return accuracy_score(expected, predicted), precision_score(expected, predicted), recall_score(expected, predicted), f1_score(expected, predicted), auc_value

        # 训练集
        y_true_last, y_pred_last = Y_train, model.predict(X_train)
        p_pred_last = model.predict_proba(X_train)
        y_scores_last = pd.DataFrame(p_pred_last, columns=model.classes_.tolist())[1].values
        acu_train = roc_auc_score(y_true_last, y_scores_last)
        print("roc_train:", acu_train)

        # 绘制ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)
        plt.figure(figsize=(10, 10))
        lw = 2
        plt.plot(fpr, tpr, color='blue', label='AUC = %0.4f' % auc_value)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def Dtree(self):
        from sklearn import tree

        X_train, X_test, Y_train, Y_test = Model.DataInit()
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, Y_train)
        predicted = clf.predict(X_test)  # 测试样本预测
        print(metrics.classification_report(Y_test, predicted))

        from sklearn.metrics import roc_auc_score
        y_true, y_pred = Y_test, clf.predict(X_test)
        y_pred_pro = clf.predict_proba(X_test)
        y_scores = pd.DataFrame(y_pred_pro, columns=clf.classes_.tolist())[1].values
        auc_value = roc_auc_score(y_true, y_scores)
        print(auc_value)
        expected = y_true
        predicted = y_pred
        return accuracy_score(expected, predicted), precision_score(expected, predicted), recall_score(expected, predicted), f1_score(expected, predicted), auc_value

        # 训练集
        y_true_last, y_pred_last = Y_train, clf.predict(X_train)
        p_pred_last = clf.predict_proba(X_train)
        y_scores_last = pd.DataFrame(p_pred_last, columns=clf.classes_.tolist())[1].values
        acu_train = roc_auc_score(y_true_last, y_scores_last)
        print("roc_train:", acu_train)

        # 绘制ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)
        plt.figure(figsize=(10, 10))
        lw = 2
        plt.plot(fpr, tpr, color='blue', label='AUC = %0.4f' % auc_value)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    # 特征重要性排序——随机森林
    def RFdemo(self):
        from sklearn.preprocessing import StandardScaler

        df = pd.read_csv("../Script/Mapping/Mfe/Sample.csv")
        cols = list(df.columns.values)
        cols_data = copy.deepcopy(cols)
        cols_data.remove('class')
        x_data = df[list(cols_data)]  # 抽取作为训练数据的各属性值
        x = np.array(x_data)
        x = StandardScaler().fit_transform(x)
        y_data = df['class']  # 最后一列作为每行对应的标签label
        y = np.array(y_data)

        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor()
        rf.fit(x, y)
        print(sorted(zip(map(lambda x:round(x, 4), rf.feature_importances_), cols_data), reverse=True))
        from matplotlib import pyplot
        pyplot.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
        pyplot.show()

    # 特征重要性排序——XGBoost
    def Lineardemo(self):
        from sklearn.preprocessing import StandardScaler

        df = pd.read_csv("../Script/Mapping/Mfe/Sample.csv")
        cols = list(df.columns.values)
        cols_data = copy.deepcopy(cols)
        cols_data.remove('class')
        x_data = df[list(cols_data)]  # 抽取作为训练数据的各属性值
        x = np.array(x_data)
        x = StandardScaler().fit_transform(x)
        y_data = df['class']  # 最后一列作为每行对应的标签label
        y = np.array(y_data)
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=4

        from xgboost import XGBClassifier
        from matplotlib import pyplot
        model = XGBClassifier(gpu_id=0)
        model.fit(x, y)
        print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), cols_data), reverse=True))
        # print(model.feature_importances_)
        pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
        pyplot.show()

    def Bysdemo(self):
        from sklearn.naive_bayes import GaussianNB

        X_train, X_test, Y_train, Y_test = Model.DataInit()
        clf = GaussianNB()
        '''
        GaussianNB 参数只有一个：先验概率priors
        MultinomialNB参数有三个：alpha是常量，一般取值1，fit_prior是否考虑先验概率，class_prior自行输入先验概率
        BernoulliNB参数有四个：前三个与MultinomialNB一样，第四个binarize 标签二值化
        这里的参数的意义主要参考https://www.cnblogs.com/pinard/p/6074222.html
        '''
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(Y_test, y_pred))
        # print(grid.best_estimator_)
        y_pred_pro = clf.predict_proba(X_test)
        y_scores = pd.DataFrame(y_pred_pro, columns=clf.classes_.tolist())[1].values
        auc_value = roc_auc_score(Y_test, y_scores)
        print("roc:", auc_value)
        Model.FigAuc(Y_test,y_scores, auc_value)


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
    modle.KnnDemo()
    # SVM
    # modle.SVMDemo()
    # RandowASVM
    # modle.RandomSvmModel()
    # modle.LogisticRegression()
    # modle.DataInit()
    # modle.Dtree()
    # 特征选择
    # modle.RFdemo()
    # modle.Lineardemo()
    # modle.Bysdemo()
    # headers = ["a", "p", "r", "f", "auc"]
    # contents = []
    # for i in range(100):
    #     content = []
    #     a, p, r, f, aucV = modle.Dtree()
    #     contents.append([a, p, r, f, aucV])
    # with open("../ML/Dtparm.csv", "w") as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerow(headers)
    #     f_csv.writerows(contents)

```
