from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score

class ModelSelection:

    def __init__(self, X_train , X_test , Y_train , Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
    
    def RandomForestClassfier(self):
        model = RandomForestClassifier(n_estimators=18) 
        model.fit(self.X_train , self.Y_train)
        Y_pred = model.predict(self.X_test)
        return Y_pred
    
    def ModelPerformance(self):
        Y_pred = self.RandomForestClassfier()
        cm=confusion_matrix(self.Y_test,Y_pred)
        #accuracy_score1=accuracy_score(self.Y_test,Y_pred)
        precision_score1=precision_score(self.Y_test,Y_pred,pos_label='positive',average='micro')
        recall_score1=recall_score(self.Y_test,Y_pred,pos_label='positive',average='micro')
        return (cm, precision_score1, recall_score1)

    def RandomForestClassfier1(self,df_sub):
        model = RandomForestClassifier(n_estimators=18) 
        model.fit(self.X_train , self.Y_train)
        Y_predsub = model.predict(df_sub)
        
        return Y_pred
    
