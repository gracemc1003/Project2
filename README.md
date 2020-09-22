# Project2

In this project, we investigate a dataset of the Shakespeare plays and the lines in the plays. We wish to create a classifier that can identify with reasonable accuracy the play which a line comes from. In this project, we heavily utilize feature engineering.

I approached this problem by addressing each possible play separately. Since the play is one of the variables we are given I created a separate dataframe for each of the plays and ran the classifier separately. I saw a significant increase in my accuracy once I did this. 

Some of the features we created in this dataset were:
* Separating the given ActSceneLine variable into three columns (one for act, one for scene, and one for line)
* Analyzing the most popular words in the dataset and creating 20 new variables counting the occurences of the 20 most used words for each individual line.

## Classification Methods
We attempted to run two different classification methods on our data: linear regression and SVC

### Linear Regression
For the linear regression model, we achieved an average accuracy of 33%. This was not ideal, however we achieved better results with our second model.

### SVC

With the SVC model we were able to reach an average accuracy of 55% over all the plays. 

## Future Directions

While I was not able to implement all the features I had in mind, if I were to work with the dataset more this is some of the possible features I could engineer:
* A variable with the number of lines since the character last spoke.
* Analyzing the acts and scenes for conversations between two characters and monologues
* Implementing vectorization of the actual string PlayerLine to get more information from the line itself. 
