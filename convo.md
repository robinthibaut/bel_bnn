# Le
Hi Robin,

After our last meeting, I worked on neural networks a lot. In my simulations, I obtained stability and temperature data for drift, and temperature for production well, from 2000 samples over a 30-year simulation with a step size of 0.1, resulting in 300 data points. Currently, I am trying to predict the stability of drift based on the first 30 to forecast the following 270.

My approach is first to reduce the dimensions of x and y using PCA within the BEL framework, then train a model using BNN to provide a predictive range for any given ytest, hoping that ytest will fall within this predicted range in each dimension.

Here are some results, showing the prior and posterior of y for a sample after PCA transformation (n=5). and the ytest, the predictions in the original space.

image.pngimage.png

The issue I am facing is that the posterior distribution is too narrow, especially for the 1-dl and 2-d data from PCA. Although the predictions seem close in the original space, the posterior range provided by the BNN often fails to include the test points.
image.png
After discussing with Thomas, we thought PBNN might perform better. What are your thoughts and suggestions?

I have pushed my code to GitHub and invited you to collaborate. Please check it 😉

The weather in Belgium is getting better. Take care of yourself in CA.
Le

# Robin
Hey Le,

Hope you're doing well! I took a deep dive into the code you shared and worked on two variant approaches that I believe addresses some of the challenges we discussed. 
I’ve outlined the key differences and adjustments below :).

• Data Preprocessing Order:
    • In your script, you applied PCA and scaling to the entire dataset before splitting into training and test sets. This leaks information from the test set into the training process, which we want to avoid. I tweaked our approach to first split the data, then apply scaling and PCA transformations. This ensures that our model is trained on data that's as "unseen" as during real-world deployment.

• Mixture Model Approach:
    • Instead of using a dense variational layer as in your BNN setup, I explored a mixture model for our predictions. This involves using a mixture of normal distributions as the output layer, which allows our model to capture a broader range of uncertainties and complexities in the data. It’s particularly useful for modeling multi-modal distributions or when we suspect the underlying data generation process isn’t well captured by a single Gaussian distribution.
    • This method also enables a more flexible representation of uncertainty, which could help with the issue of narrow posterior distributions you encountered. By using a mixture of normals, we can potentially capture wider and more realistic predictive intervals.

 • Model Training and Validation:
    • I added an early stopping mechanism to prevent overfitting and to halt training when the validation loss stops improving. This helps in conserving resources and prevents the model from learning noise in the training data.

However, I still have some doubt about the classic neural network here. For time series predictions, we’d better use an LSTM. I’ve drafted an example that includes dropout to estimate the uncertainty, although this is far from perfect, I’d like to use a distribution mixture as the output layer, but I’m running out of time for today :p.

My codes are located in https://github.com/robinthibaut/bel_bnn/tree/master/bnn. I’m sure you’ll have no troubles to adapt them.

Let me know if you have any questions or if you’d like to set a meeting :).
The weather here is getting amazing again! 
Take care,
Robin

# Le
Hi Robin,

Thanks for the reply! I've pulled the code and run it, and it looks like the performance of both MixtureNormal and LSTM is quite impressive. However, I haven't fully grasped all the details yet. I will make sure to fully understand these aspects ASAP and update you with my thoughts. 😉

Best,
Le

# Robin
Hi Le,

Take your time :). By the way, I finished reading your draft on overleaf and added some comments, can you see them? I think the manuscript is really well-written, and the figures are splendid!

Robin

# Le
Hi Robin,

Thanks for the reply! I've pulled the code and run it, and it looks like the performance of both MixtureNormal and LSTM is quite impressive. However, I haven't fully grasped all the details yet. I will make sure to fully understand these aspects ASAP and update you with my thoughts. 😉

Best,
Le

# Robin
Hi Le,

I think too that PNNs are the best options for BEL.

In a nutshell, "BEL" or Bayesian Simulation-Informed Learning - which I believe is a more appropriate name - simply means training a machine learning algorithm with uncertainty quantification at the end of the pipeline. As a result, the method used to connect the inputs and outputs makes no difference. The target's conditional posterior probability distribution can be expressed using a variety of methods, including CCA, neural networks...

I'd like to discuss one important aspect of your problem. I believe it would be difficult to justify not using a recurrent neural network (RNN), such as LSTM, for time series prediction. This family of architecture is designed to handle time series, whereas the traditional approach (feedforward neural networks - FNN) is not. When using RNNs, a minor issue may arise, i.e. how to incorporate uncertainty quantification. For feedforward neural networks, such as the MDN I sent you, we directly fit a mixture of Gaussians to the entire time series. I found it difficult to include such an output layer in the LSTM, but it’s surely possible with tuning. Some authors have done this before, for example, https://hess.copernicus.org/articles/26/1673/2022/. Note that if we use RNNS, we do not need to apply PCA anymore.

I think that it would be worth investigating the use of an LSTM with a posterior distribution estimation at the end of the pipeline. 

EDIT: While writing this email, I was playing around with neural networks and managed to plug an MDN directly after the LSTM. I’m happy with the initial results, but it definitely needs improvements. You can find it in robin_lstm_2.py.

Let me know what you think :).

Cheers,
Robin

# Thomas (PI)
Hi both,

Interesting discussion.

It is indeed worth investigating the LSTM. Nevertheless, I am less convinced than Robin that it might be a requirement. I might be wrong, but as I understand, LSTM have been mostly developed for fully-driven approach. For example, if you predict a surface water catchment, they could predict the evolution of a flood based on the early increase in water level or the precipitation record. In essence, they are made for highly transient systems, aren't they? What Le is currently doing, although transient, is easily predictable from the simulation-based training set. PCA is efficient in capturing that trend in the evolution of the target.

But anyway, in an LSTM is working, then it can be a good innovation. Indeed, eventually, a real risk-prediction system would be much more transient than what we are now simulating, with flowrate that are not constant (see example of Luka for an ATES system).

The most important think is to maintain indeed the uncertainty in the prediction!

Best,
Thomas

# Le
Hi Robin,

Good Morning! Here are some updates.

I have some new ideas and results I'd like to share with you. I found that in some papers, the data post-PCA can also be input into an LSTM although you said that if we use RNNS, we do not need to apply PCA anymore. :) , I try to combine the LSTM+MDN model with the PCA+MDN model we discussed before, establishing a PCA+LSTM+MDN model. and test the model's performance. For detailed model configuration and results, please check the Word document and my GitHub.

Because PCA is applied, the prediction results are smoother in this model, and it allows for an intuitive observation of the posterior distribution of each PC. Moreover, the application of LSTM captures the evolutionary trend of predictions well, even though the dimensionality reduction reduces y from 270 dimensions to 3 dimensions while retaining 99% of the information.

In total, four models are tested: PLDM (PCA-LSTM-Dropout-MDN), PLM (PCA-LSTM-MDN), LDM (LSTM-Dropout-MDN), and PM (PCA-MDN). For each model, hyperparameters are first tested using Keras Tuner, and then evaluations are conducted using CRPS (Continuous Ranked Probability Score), RMSE, and RMSPE. Overall, the results of PLDM and PLM are the most ideal.

Do you think adding PCA first is a good option and this method could serve as an alternative to CCA?

The Word link:
update.docx​

Best,
Le

# Robin
Hi Le and Thomas,

Thanks for sharing your thoughts and updates. Let me address both of your concerns and ideas.

To Le:

Regarding the application of LSTM models for our project, the decision to advocate for LSTM usage (or RNN) primarily stems from its design to capture sequential dependencies in time-series data effectively. The nature of your project, which involves predicting stability over time based on historical data, inherently suits the capabilities of LSTM models. These models are built to understand and leverage the temporal dynamics inherent in time-series data, which could significantly enhance our prediction accuracy and reliability for future states based on past and present observations.

LSTMs, unlike traditional neural networks, are capable of remembering information for an extended period, making them particularly suitable for our project where past data points influence future predictions. This capability is crucial for capturing the temporal patterns and dependencies in the stability data you're working with.

I have a different ideas, though, about how to use Principal Component Analysis (PCA) before LSTM. PCA, while beneficial for dimensionality reduction and highlighting the most significant features, may “obscure" the temporal relationships that LSTMs are designed to capture. By transforming the original data into principal components, we risk losing some of the sequential information that is crucial for LSTM models to make accurate predictions. This is because PCA focuses on the variance in the dataset without considering the temporal order of the data, which is a vital component of time-series analysis.

However, if you found something in the literature that contradicts this, please share it with me, I’m highly interested! Your approach of integrating PCA with LSTM in a hybrid model is innovative and could potentially yield benefits in terms of computational efficiency and model performance. The success of such a model would largely depend on the specific characteristics of your data and how well the reduced dimensions preserve the temporal dependencies.

To Thomas:

I understand that you don't think LSTM is essential when compared to PCA or the original neural network method. LSTMs are indeed more commonly associated with highly transient systems where the sequence of data points plays a critical role in prediction accuracy. However, the decision to consider LSTM for our project is based on the premise that even subtle temporal patterns within the stability data could significantly impact prediction outcomes. While the system may appear predictably transient with a simulation-based training set, LSTMs offer the potential to uncover and leverage deeper temporal insights that might be missed by other models.

To add to my arguments for sequential models, I’ve been looking for a metaphor… Using a non-sequential model for time series data is like trying to screw in a bolt with a hammer. You might apply a lot of effort, but without the fit of a wrench, you won't be able to turn the bolt effectively :p. Anyway, if we decide to use a “classical” neural network, it could potentially be fine too, but we’d have to justify it in the paper.

I would also like to add a point about time series predictions the “BEL” way: when using models for time series prediction, we expect the early stages of the predicted time series to exhibit lower variance. This expectation stems from the fact that predictions made closer to the last known data point have more immediate prior information to draw upon, leading to higher confidence and lower uncertainty in these early predictions. The model can leverage recent, more relevant data, which typically results in predictions that are more precise and less varied. That’s why I think the “optimal” architecture should produce such predictions, and would be a criterion in our model selection.

I hope these insights help clarify the rationale behind considering LSTM models for our project. The results you’ve shared so far are promising, and I look forward to discussing them further.
