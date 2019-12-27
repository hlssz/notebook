   
   ``` bash
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
```
