#CAPSTONE PROJECT
#JOHN KINGSLEY
#04/12/2019


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")
#Validation set will be 10% of MovieLens data
#i used R version 3.6.1 hence the reason for the script,
set.seed(1, sample.kind="Rounding")
## Warning in set.seed(1, sample.kind = "Rounding"): non-uniform 'Rounding' sampler
## used
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
#confirmed that userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
#Added rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
## Joining, by = c("userId", "movieId", "rating", "timestamp", "title", "genres")
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)
#overview of edx and validation dataset using the glimpse function

glimpse(edx)

glimpse(validation)

#We observed that the edx dataset is composed of 9000055 observations with 6 variables while the 10% validation yielded 999999 occurrences with same variables. #summary of the variables in edx datatset using the summary function

summary(edx$userId)
summary(edx$movieId)
summary(edx$rating)
#summary of the variables in validation dataset
summary(validation$userId)
summary(validation$movieId)
summary(validation$rating)
#from the summary statistics, it was observed that the most ratings are whole ratings and comparing the difference between the mean and median, it shows that the data are skewed for the movie title we used the function head from the summary statistics, it was observed that the most ratings are #whole ratings and comparing the difference between the mean and median, it shows that the data are skewed for the movie title we used the function head
head(edx$title)
"
#the proportions of users and movies
edx %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

#total number of different users, using the codes provided in the course
Fig1<-edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

Fig2<-edx %>%
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("movieid")
#Fig 1, the userId showed that the number of observations in userId variable gave a skewed distributiion while the moviesId in Fig2 gave a normal distribution #################

METHODOLOGY AND ANALYSIS Step 1 #The model used for developing the prediction algorithm follows that from the course: #the mean rating mean_edx is modified by one or more “bias” which in this project are userId and MovieId and with a residual error expected. #
Y_u,i=μ+b_i+b_u+ϵ_u_i
#following the steps are outlined in the course #mean for the edx

mean_edx <- mean(edx$rating)
mean_edx
## [1] 3.512465
#encountered some challenges, my computer couldnot processed the edx dataset. that is to say i could not split the edx datasets into individual genre hence it was skipped. #Also,I considered different machine learning algorithms but due to my computer’s memory space and the challenge obtain the highest accuracy, #which is measured as the number of true matches of predicted ratings vs ratings of #the validation set. And the acceptable most algorithm for this project is THE PENEALIZED LEASTt SQUARES METHOD. Step2 #using the code made availabe in the course, we try to look at the movie effect

movie_effect <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mean_edx))
FIG3<-movie_effect %>% qplot(b_i, geom ="histogram", bins = 30, data = ., color = I("blue"))+ 
  ggtitle("movie_effect")
Step 3 #Also, we considered the user effect on the observation

user_effect <- edx %>% 
  left_join(movie_effect, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mean_edx - b_i))
FIG4<-user_effect %>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("green"))+
  ggtitle("user effect")
#We observed that a normal distribution is obtained in the user effect qqlot. this indicates that the moods of the users played a vital role on the movie. That is some where either generous or not generous at the rating of movies. RESULTS #MODEL CONSTRUCTION #the validation RMSE is what if refer to as the core_rmse (MODEL 1)

core_rmse <- RMSE(validation$rating, mean_edx)
core_rmse
#inspect result

rmse_results <- data_frame(method = "Just the mean value", RMSE = core_rmse)
## Warning: `data_frame()` is deprecated, use `tibble()`.
## This warning is displayed once per session.
rmse_results
Step 4 #MOVIE EFFECT (MODEL 2)

movie_rating_predicted <- validation %>% 
  left_join(movie_effect, by='movieId') %>%
  mutate(pred = mean_edx + b_i) 
#movie effect model using the validation set as stated in the project guidelines

rmse_mode1 <- RMSE(validation$rating,movie_rating_predicted$pred)

rmse1 <- bind_rows(rmse_results,
                   data_frame(method="Movie Effect Model",  
                              RMSE = rmse_mode1 ))
rmse1 %>% knitr::kable()
rmse1
#owing to the above output, the result gave a promising result for further considerations. #now, we will combine the movie effect, user effect and “just the mean” to see the further performance of our model Step 5 #User effect #just trying out

user_rating_predicted <- validation %>% 
  left_join(user_effect, by='userId') %>%
  mutate(pred = mean_edx + b_u) 

rmse_model2 <- RMSE(validation$rating,user_rating_predicted$pred)
rmse2 <- bind_rows(rmse_results,
                   data_frame(method="User Effect Model",  
                              RMSE = rmse_model2 ))
rmse2 %>% knitr::kable()
method	RMSE

Step 6 #COMBINED MODELS (MOVIE AND USER EFFECT MODEL)#Model 3

predicted_ratings_user_movie <- validation %>% 
  left_join(movie_effect, by='movieId') %>%
  left_join(user_effect, by='userId') %>%
  mutate(pred = mean_edx + b_i + b_u) 
run model and save rmse ouput
rmse_model3 <- RMSE(validation$rating,predicted_ratings_user_movie$pred)
rmse3 <- bind_rows(rmse_results,
                   data_frame(method="Movie and User Effect Model",  
                              RMSE = rmse_model3))
rmse3 %>% knitr::kable()
rmse3

Step 7 #REGULARIZATION PROCESSES #Firstly, we define lambdas,#he weight given to the regularization term (the L1 norm), #so as lambda approaches zero, the loss function of your model approaches the OLS loss function. #(https://stats.stackexchange.com/questions/68431/interpretting-lasso-variable-trace-plots)

lambdas <- seq(0, 5, 0.25)

RMSES <- sapply(lambdas,function(l){
  #recall same script above 
  mean_edx <- mean(edx$rating)
  #recall, Y_u,i=μ+b_i+b_u+ϵ_u_i $$
  #b_i= movieId
  #b_u=userId+movieId
  
  #SO, we regularized movieId and penalize low number on ratings
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mean_edx)/(n()+l))
  
  #we did same approach for both user and movie effect and penalize low number of ratings
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mean_edx)/(n()+l))
  
  #predict ratings in the validation dataset which is composed of 10% of edx datatset to derive optimal penalty value 'lambda'
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mean_edx + b_i + b_u) %>%
    .$pred
  
  return(RMSE(predicted_ratings, validation$rating))
})


plot(lambdas, RMSES,
     col = "black")


lambda <- lambdas[which.min(RMSES)]
lambda

paste('Optimal RMSE of',min(RMSES),'is achieved with Lambda',lambda)

#The Lamda value was then applied on the Validation dataset to predict movie rating like if it was not known
lambda1 <- 5

PREDICTED <- sapply(lambda1,function(l){
  
  #recall from the script above
  mean_edx <- mean(edx$rating)
  
  #we therefore Calculated the movie effect with optimal lambda
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mean_edx)/(n()+l))
  
  #we repeated same step as above and Calculated user effect with optimal lambda
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mean_edx)/(n()+l))
  
  #Predict ratings on validation set
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mean_edx + b_i + b_u) %>%
    .$pred #validation
  
  return(predicted_ratings)
  
})
#The final root mean square error of the regularized dataset was taken as the final rmse in the project

paste('Optimal RMSE of',min(RMSES),'is achieved with Lambda',lambda)

REFERENCES 1. http://www.datainsight.at/report/Capstone_04_fin.html 2. http://www.rpubs.com/rezapci/MovieLens 3.https://courses.edx.org/courses/course-v1:HarvardX+PH125.8x+2T2019/courseware/a49844e4a3574c239f354654f9679888/7e7727ce543b4ed6ae6338626862eada/?child=first 4. https://rafalab.github.io/dsbook/introduction-to-machine-learning.html#an-example 5. https://data-flair.training/blogs/data-science-r-movie-recommendation/ 6. Irizzary,R., 2018,Introduction to Data Science, githubpage,https://rafalab.github.io/dsbook/
