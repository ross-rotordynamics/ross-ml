# ross-ml

# ross-ml

Pipeline: 

1) Q: What is a Pipeline? 
   
   A: "Is a set of data processing elements connected in series, where the output of one element is the input of the next one"
2) Q: How to use this Pipeline?
   
   A: First of all, you should pass to the Pipeline class a DataFrame, ie, a structured data format from pandas. This is quite easy to obtain if your file results are a csv
   or a xls file. In order to do that, use the comand pandas.read_csv('your_file.csv/xls') etc.
   After obtaining a DataFrame file, pass it to the Pipeline by using the command: 
   
   D = pipeline(df)
   
   this will instantiate a pipeline object. After that, you must inform the program which columns will be your features and which ones will be your labels. To do that, use the
   commands: D.set_features(begin,end) and D.set_labels(begin,end). Quite self-explanatory, isn't it?
   
   Ok. Now you have informed the program about your labels and features. What's next? The next step is to use a scalers in your input, output or both. There are plenty of them
   on scikit-learn. Give yourself a time to read it: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html. , since your results depends on how well 
   you scale your data. To scale your data, you should use the command: data_scaling(test_size,scalers = [scaler1,scaler2], scaling = True). test_size variable tells the program
   how great will be your cross validation data in %. In Machine Learning communities, 30% is considered a good amount of data for cross-validation. scalers are a list with the
   input scaler and the output one, respectively.
   
   After proper scaling, it is time to build your Artificial Neural Networks (ANN). For this, use the following command: D.build_Sequential_ANN(number_of_layers,neurons_per_layer).
   It is also possible to add dropout layers on your model by using:D.build_Sequential_ANN(number_of_layers,neurons_per_layer,dropout_layers,dropout_values). number_of_layers tells
   the program how many layers will be used,while neurons_per_layer tells how many neurons will be used in each layer. So, it is clear that neurons_per_layer is a list data structure.
   
   For dropout insertion, the idea is analogous: you need to inform the program the dropout layer positions and how much dropout you will add to those layers. So, both 
   dropout_layers and dropout_values are lists.
   The next step is to run your model. To do that use the comand D.run_model(optimizer,loss,batch_size,epochs). optimizer inform the program which optmizer will be used,loss is
   the loss function that will be used in optmization process, batch_size tells the program how many batches will be used and epochs informs the how many epochs will take the
   training. The syntax here is the same used on Keras.
   
   Now, it's time to check your results. Firstly, you should check if your model presents overfit. Overfit,in simple terms, is when model focus more on fitting the given points 
   than recognizing the "big picture" of your data. Because of that, overfited models suffers from generalization. To check your model, use D.model_history(). This will plot
   a graph of both val_loss and loss. To a better understanding what each one means, please check https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/62261 or
   cross-validation forums.
   
   If after checking your loss, if you want to quantify how good your model is, it is recommended,after using D.validation() ,to use D.metrics(),which returns several metrics commonly used on Machine Learning
   community. Here, a piece of advice is needed: the metrics used varies according to your problem. The most straightfoward way to choose a proper metric is to check if your 
   understanding of problem is adequate and then rethink the problem if it is necessary. It is also recommended to check if anyone else has the same problem has the same problem as 
   you, what is the most common scenario. I personaly recommend https://stats.stackexchange.com/ and Quora forums.
   
   It is also provided hypothesis tests to check your model. However, you should check if the following tests has enough power to be conclusive. Here, it is proposed two
   hypothesis test: ks-test and welch test. The former tries to verify if both test and train comes from the same distribution, which could mean that both are samples from
   the same distribution which describes your phenomenon. The last tries to check if the trained model has the same mean as the test set, which indicate if your model are 
   accurate or not. Again, you should get the power of both hypothesis to properly make conclusions out of hypothesis tests.
   
   If your model is adequate, you can save it using the command D.save_model().
   
   3) Q: Is there a way to visualize all the results?
   
      A: Yes! There is a postprocessing class to that purpose, in which you inform your trained data and validation one. To do so you should use 
      postproc = postprocessing(D.train,D.test), where D.train and D.test are obtaining after instatiating pipeline class. After that you are able to plot the following graphics:
      - q-q plot
      - confidence intervals of your predictions, using DKW-inequality
      -standardized error plot 
      - resume of residuals.
      
     4) Q: Is it possible to reuse the model?
     
        A: Surely! You can load your model by using the following commands: model = Model('filename') and model.load_model(). The first command searches for the saved file while
        the second loads it. To make predictions with your loaded model, use model.predict(X), where X are the features that you want to predict. Make sure that X has the same
        shape that your former features in order to guarantee proper results
   
   
   
   

   
