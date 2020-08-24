# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:10:24 2020

@author: Luan
"""
import os
import webbrowser
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,PowerTransformer,QuantileTransformer,Normalizer,MaxAbsScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score,r2_score
from scipy.stats import ks_2samp,entropy,chisquare,ttest_1samp,ttest_ind,skew,normaltest
from keras.regularizers import l2
from sklearn.feature_selection import SelectKBest,mutual_info_regression,f_regression
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.decomposition import PCA
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.tree import DecisionTreeRegressor
from pickle import dump,load

def HTML_formater(df,name):
    html_string = '''
                    <html>
                    <head><title>HTML Pandas Dataframe with CSS</title></head>
                    <link rel="stylesheet" type="text/css" href="css/panda_style.css"/>
                   <body>
                       {table}
                   </body>
                    </html>.
               '''
    with open('tables/{}.html'.format(name), 'w') as f:
        f.write(html_string.format(table=df.to_html(classes='mystyle')))



class pipeline:
    def __init__(self,df):
        path1 = 'img'
        path2 = 'tables'      
        if not os.path.isdir(path1):
            os.makedirs(path1)
        if not os.path.isdir(path2):
            os.makedirs(path2)
        self.df = df
        self.df.dropna(inplace = True)
    
    def set_features(self,begin,end):
        '''
        

        Parameters
        ----------
        begin : int
            begin column of dataframe features
        end : TYPE
            end column of dataframe features

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        self.X = self.df[self.df.columns[begin:end]]
        return self.X
    
    def set_labels(self,begin,end):
        '''
        

        Parameters
        ----------
        begin : TYPE
            begin column of dataframe labels
        end : TYPE
            end column of dataframe labels


        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        self.y = self.df[self.df.columns[begin:end]]
        return self.y
   
    def feature_reduction(self,N_top):
        '''
        

        Parameters
        ----------
        N_top : int
            Number of relevant features

        Returns
        -------
        TYPE
            Minimum number of features that satisfies N_top for each label

        '''
        # define the model
        model = DecisionTreeRegressor()
        # fit the model
        model.fit(self.X, self.y)
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        featureScores = pd.concat([pd.DataFrame(self.X.columns),pd.DataFrame(importance)],axis=1)
        featureScores.columns = ['Specs','Score']
        self.best = featureScores.nlargest(N_top,'Score')['Specs'].values
        # col_x = self.X.columns
        # col_y = self.y.columns
        # best_features = []
        # X = self.df[col_x]
        # for var in col_y:
        #     y = self.df[col_y][var]
        #     bestfeatures = SelectKBest(score_func=mutual_info_regression, k=N_top)#bestfeatures = SelectKBest(score_func=f_regression, k=N_top)
        #     fit = bestfeatures.fit(X,y)
        #     #dfscores = pd.DataFrame(fit.scores_)
        #     #dfcolumns = pd.DataFrame(X.columns)
        #     featureScores = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(fit.scores_)],axis=1)
        #     featureScores.columns = ['Specs','Score']  #naming the dataframe columns
        #     best_features.append(featureScores.nlargest(N_top,'Score')['Specs'].values)
        # best_features = np.unique(np.array(best_features))
        #self.X = self.X[best_features]
        
        self.X = self.X[self.best]
        return self.X
    
    def data_scaling(self,test_size,scaling = True,scalers = []):
        '''
        

        Parameters
        ----------
        test_size : float
            percentage of data destined for testing
        scalers : scikit-learn object
            scikit-learn scalers
        scaling : boolean, optional
            Choose between scaling the data or not. The default is True.

        Returns
        -------
        Array
            X_train - features destined for training.
        Array
             X_test -features destined for test.
        Array
            y_train - labels destined for training.
        Array
             y_test - labels destined for test.

        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size = test_size)
        if scaling:
            if len(scalers) >= 1:
                self.scaler1 = scalers[0]
                self.X_train = self.scaler1.fit_transform(self.X_train)
                self.X_test = self.scaler1.transform(self.X_test)
                if len(scalers) == 2:
                    self.scaler2 = scalers[1]
                    self.y_train = self.scaler2.fit_transform(self.y_train)
                    self.y_test = self.scaler2.transform(self.y_test)
        else:
            self.scaler1 = None
            self.scaler2 = None
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_Sequential_ANN(self,hidden,neurons,dropout_layers = [],dropout = []):
        '''
        

        Parameters
        ----------
        hidden : int
            Number of hidden layers.
        neurons : list
            Number of neurons per layer.
        dropout_layers : list, optional
            DESCRIPTION. dropout layers position.The default is [].
        dropout : list, optional
            DESCRIPTION. list with dropout values. The default is [].

        Returns
        -------
        keras neural network

        '''
        self.model = Sequential()
        self.model.add(Dense(len(self.X.columns),activation='relu'))
        j = 0 # Dropout counter
        for i in range(hidden):
            if i in dropout_layers:
                self.model.add(Dropout(dropout[j]))
                j+=1
            self.model.add(Dense(neurons[i],activation='relu'))
        self.model.add(Dense(len(self.y.columns)))
        self.config = self.model.get_config()
        return self.model
    
    def model_run(self,optimizer = 'adam',loss = 'mse',batch_size = 16,epochs = 500):
        '''
        

        Parameters
        ----------
        optimizer : string, optional
            Choose a optimizer. The default is 'adam'.
        loss : string, optional
            Choose a loss. The default is 'mse'.
        batch_size : int, optional
            batch_size . The default is 16.
        epochs : int, optional
            Choose number of epochs. The default is 500.

        Returns
        -------
        object
            model
        object
            predictions

        '''
        self.model.compile(optimizer= optimizer,loss = loss)
        self.history = self.model.fit(x=self.X_train,y=self.y_train,
              validation_data=(self.X_test,self.y_test),
              batch_size=batch_size,epochs= epochs)# batch_size = 16,epochs = 500
        self.predictions = self.model.predict(self.X_test)
        self.train = pd.DataFrame(self.scaler2.inverse_transform(self.predictions),columns =self.y.columns)
        self.test = pd.DataFrame(self.scaler2.inverse_transform(self.y_test),columns = self.y.columns)
        return self.model,self.predictions
    def model_history(self):
        '''
        Plot model history

        Returns
        -------
        None.

        '''
        hist = pd.DataFrame(self.history.history) #hist = pd.DataFrame(self.model.history.history)
        plt.plot(np.log10(hist['loss']),label='loss')
        plt.plot(np.log10(hist['val_loss']),label = 'val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('$log_{10}(Perda)$')
        plt.title('Loss Function')
        plt.legend()
        plt.savefig('img\history.png', dpi=1000, facecolor='w', edgecolor='k',
        orientation='portrait',format='png')
    def metrics(self):
        '''
        Print Metrics

        Returns
        -------
        None.

        '''
        R2_a = 1 - ((len(self.predictions)-1)*(1-r2_score(self.y_test,self.predictions))/(len(self.predictions)-(1+len(self.X.columns))))
        MAE = mean_absolute_error(self.y_test,self.predictions)
        MSE = mean_squared_error(self.y_test,self.predictions)
        R2 = r2_score(self.y_test,self.predictions)
        explained_variance = explained_variance_score(self.y_test,self.predictions)
        metrics = pd.DataFrame(np.round([MAE,MSE,R2,R2_a,explained_variance],3),index = ['MAE','MSE','R2','R2_adj','explained variance'],columns = ['Metric'])
        #metrics.to_html('Metrics.html')
        HTML_formater(metrics, 'Metrics')
        print('Scores:\nMAE: {}\nMSE: {}\nR^2:{}\nR^2 ajustado:{}\nExplained variance:{}'.format(MAE,MSE,R2,R2_a,explained_variance))
    def hypothesis_test(self,kind = 'Komolgorov-Smirnov',p_value = 0.05):
        '''
        

        Parameters
        ----------
        kind : string, optional
            Hypothesis test kind. Choose between 'Welch' and K-S tests. The default is 'Komolgorov-Smirnov'.
        p_value : float, optional
            Critical value.A number between 0 and 1. The default is 0.05.

        Returns
        -------
        p_df : DataFrame
            Hypothesis test results.

        '''
        if kind == 'Komolgorov-Smirnov':
            p_values = np.round([ks_2samp(self.train[var],self.test[var]).pvalue for var in self.train.columns],3)
            p_df = pd.DataFrame(p_values,index = self.test.columns,columns = ['p-value: train'])
            p_df['status'] = ['Not Reject H0' if p>p_value else 'Reject H0' for p in p_df['p-value: train']]
            HTML_formater(p_df, 'KS Test')
            #p_df.to_html(r'tables\K-S test.html')
        if kind == 'Welch':
            p_values3 = np.round([ttest_ind(self.train[var],self.test[var],equal_var=False).pvalue for var in self.train.columns],3)
            p_df = pd.DataFrame(p_values3,index = self.train.columns,columns = ['p-value: acc'])
            p_df['status'] = ['Not Reject H0' if p>p_value else 'Reject H0' for p in p_df['p-value: acc']]
            HTML_formater(p_df, 'Welch Test')
            #p_df.to_html(r'tables\Welch test.html')
        return p_df
    def validation(self,X,y):
        self.test = y
        if self.scaler1 != None:
            X_scaled = self.scaler1.transform(X.values.reshape(-1,len(X.columns)))
        if self.scaler2 != None:
            self.train = pd.DataFrame(self.scaler2.inverse_transform(model.predict(X_scaled)),columns = y.columns)
        else:
            self.train = pd.DataFrame(model.predict(X_scaled),columns = y.columns)
        return self.train,self.test
    def save_model(self,name):
        if not os.path.isdir(name):
            os.makedirs(name)
        self.model.save(r'{}/{}.h5'.format(name,name))
        dump(self.y.columns,open(r'{}/{}_columns.pkl'.format(name,name),'wb'))
        dump(self.best,open(r'{}/{}_best_features.pkl'.format(name,name),'wb'))
        if self.scaler1 != None:
            dump(self.scaler1, open(r'{}/{}_scaler1.pkl'.format(name,name), 'wb'))
        if self.scaler2 != None:
            dump(self.scaler2, open(r'{}/{}_scaler2.pkl'.format(name,name), 'wb'))
            
                
class Model():
    def __init__(self,name):
        self.name = name
    def load_model(self):
        self.model = load_model(r'{}/{}.h5'.format(self.name,self.name))
        self.columns = load(open(r'{}/{}_columns.pkl'.format(self.name,self.name), 'rb'))
        #load best features
        try:
            self.best = load(open(r'{}/{}_best_features.pkl'.format(self.name,self.name),'rb'))
        except:
            self.best = None
        # load the scaler
        try:
            self.scaler1 = load(open(r'{}/{}_scaler1.pkl'.format(self.name,self.name), 'rb'))
        except:
            self.scaler1 = None
        try:
            self.scaler2 = load(open(r'{}/{}_scaler2.pkl'.format(self.name,self.name), 'rb'))
        except:
            self.scaler2 = None
        #return self.model
    def predict(self,X):
        if self.best is None:
            self.X = X
        else:
            self.X = X[self.best]
        if self.scaler1!= None:
            self.X = self.scaler1.transform(self.X)
        else:
            self.X = X
        if self.scaler2!= None:
            predictions = self.model.predict(self.X)
            self.results = pd.DataFrame(self.scaler2.inverse_transform(predictions),columns = self.columns)
        else:
            self.results =  pd.DataFrame(predictions,columns = self.columns)
        return self.results
            
        
        
    
    
    
    
    
class postprocessing():
    def __init__(self,train,test):
        self.train = train
        self.test = test
        self.test.index = self.train.index
    def show_overall_results(self):
        df = pd.concat([self.train,self.test],axis=0)
        df['label'] = ['train' if x<len(self.train) else 'test' for x in range(len(df))]
        sns.pairplot(df,hue = 'label')
        plt.savefig(r'img\pairplot.png', dpi=1000, facecolor='w', edgecolor='k',
        orientation='portrait',format='png')
        plt.close()
    def show_confidence_bounds(self,a):
        '''
        Plot a Confidence interval based on DKW inequality

        Parameters
        ----------
        a : float
            Significance level.A number between 0 and 1.

        Returns
        -------
        None.

        '''
        sns.set()
        P = np.arange(0,100,0.05)
        en = np.sqrt((1/(2*len(P)))*np.log(2/a))
        var = list(self.train.columns)#[0:8]
        #figure, axes = plt.subplots(nrows=4, ncols=2)
        for v in var:
            plt.figure()
            F = ECDF(self.train[v])
            G = ECDF(self.test[v])
            Gu = [min(Gn+en,1) for Gn in G.y]
            Gl = [max(Gn-en,0) for Gn in G.y]
            plt.plot(G.x,Gu,label = '{} - 95% CI'.format(v),ls = '--',color = 'black')
            plt.plot(G.x,Gl,ls = '--',color = 'black')
            plt.plot(G.x,G.y,label = ' test {}'.format(v),ls = '-')
            plt.plot(F.x,F.y,label = 'train {}'.format(v),ls = ':',linewidth = 2,color = 'magenta')
            plt.legend()
            plt.xscale('symlog')
            plt.savefig(r'img\CI_{}.png'.format(v), dpi=1000, facecolor='w', edgecolor='k',
        orientation='portrait',format='png')
            plt.close()
        
    def q_q_plot(self):
        var = list(self.train.columns)#[0:4]
        for v in var:
            plt.figure()
            plt.scatter(self.test[v],self.train[v],color='orange',label = 'pontos de teste - {}'.format(v))
            plt.plot(self.test[v],self.test[v],'b',label = '$y_{teste}=y_{treino}$')
            plt.legend()
            plt.savefig(r'img\qq_plot_{}.png'.format(v), dpi=1000, facecolor='w', edgecolor='k',
        orientation='portrait',format='png')
            plt.close()
    def standardized_error_plot(self):
        var = list(self.train.columns)#[0:8]
        erro = self.test-self.train
        erro = erro/erro.std()
        erro.dropna(inplace = True)
        for v in var:
            plt.figure()
            #erro = self.test[v]-self.train[v]
            plt.scatter(self.test[v],erro[v],color='orange',label = 'pontos de teste - {}'.format(v))
            plt.axhline(0, ls='--',color = 'blue')#plot(self.test[var[i]],np.zeros(len(test[var[i]])),'--b',label = r'$\sigma$=0')
            plt.legend()
            #plt.title()
            plt.savefig(r'img\standardized_error_{}_plot.png'.format(v), dpi=1000, facecolor='w', edgecolor='k',
        orientation='portrait',format='png')
            plt.close()
           
    def residuals_resume(self):
        erro_std = ((self.test-self.train)/(self.test-self.train).std())
        plt.figure(figsize=(16,9))
        g = sns.boxenplot(data = erro_std,orient="h", palette="hls")#,showfliers = False)
        g = sns.stripplot(data = erro_std,orient = 'h',palette = 'hls')
        ax1 = g.axes
        ax1.axvline(0, ls='--',color = 'black',label = r'$\sigma$ = 0')
        ax1.axvline(1, ls='--',color = 'red',label = r'$\sigma$ = 1')
        ax1.axvline(-1, ls='--',color = 'red')
        ax1.axvline(2, ls='--',color = 'blue',label = r'$\sigma$ = 2')
        ax1.axvline(-2, ls='--',color = 'blue')
        ax1.axvline(3, ls='--',color = 'green',label = r'$\sigma$ = 3')
        ax1.axvline(-3, ls='--',color = 'green')
        plt.title('Distribuição dos resíduos')
        plt.xlim([-10,10])
        plt.legend()
        plt.savefig('img\residuals_resume.png', dpi=1000, facecolor='w', edgecolor='k',
        orientation='portrait',format='png')
    @staticmethod
    def show(url = 'results.html'):
        return webbrowser.open(url,new=2)
        
    
    
    
        
        
    





               
                
    
# Example: 

#df = pd.read_csv('seal_fake.csv')#('xllaby_data-componentes.csv')
#df_val= pd.read_csv('xllaby_data-componentes.csv')
#df_val.fillna(df.mean)
#df.drop(['label'],axis = 1,inplace = True)
#D = pipeline(df)
#D.set_features(0,20)
#D.set_labels(20,len(D.df.columns))
#D.feature_reduction(15)
#D.data_scaling(0.1,scalers = [RobustScaler(),RobustScaler()], scaling = True)
#D.build_Sequential_ANN(4,[50,50,50,50])#,dropout_layers = [1,2,3],dropout = [0.001,0.001,0.001])
#model,predictions = D.model_run(batch_size = 300,epochs =1000) 
#model.get_config()
#D.model_history()
#D.metrics()
#postproc = postprocessing(D.train,D.test)
#postproc.show_overall_results()
#postproc.show_confidence_bounds(a = 0.01)
#postproc.standardized_error_plot()
#postproc.q_q_plot()
#postproc.show()
#url = 'results.html'
#D.hypothesis_test()
#D.save_model('teste')
#model = Model('teste')
#model.load_model()
#X= pipeline(df_val).set_features(0,20)
#results = model.predict(X)

