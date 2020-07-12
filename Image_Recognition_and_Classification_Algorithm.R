# Nitin Rai
# Agricultural and Biosystems Engineering
# North Dakota State University
#Load Libraries, i.e., EBImage and Keras (Using Tensorflow by Google as Backend)
library(png)
library(EBImage)
library(keras)
#install_tensorflow()
#install_tensorflow(gpu=TRUE) #for GPU support
#install_tensorflow(
  #method = c("auto", "virtualenv", "conda"),
  #conda = "auto",
  #version = "default",
  #envname = NULL,
  #extra_packages = NULL,
  #restart_session = TRUE,
  #conda_python_version = "3.6",
#)
library(tensorflow)
# for wheat detection 
#pics <- c('1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg', '14.jpg', '15.jpg', '16.jpg', '17.jpg', '18.jpg', '19.jpg', '20.jpg', '21.jpg', '22.jpg', '23.jpg', '24.jpg', '25.jpg',
          #'26.jpg', '27.jpg', '28.jpg', '29.jpg', '30.jpg', '31.jpg', '32.jpg', '33.jpg', '34.jpg', '35.jpg', '36.jpg', '37.jpg', '38.jpg', '39.jpg', '40.jpg', '41.jpg', '42.jpg', '43.jpg', '44.jpg', '45.jpg', '46.jpg', '47.jpg', '48.jpg', '49.jpg', '50.jpg',
          #'t1.jpg', 't2.jpg', 't3.jpg', 't4.jpg', 't5.jpg', 't6.jpg', 't7.jpg', 't8.jpg', 't9.jpg', 't10.jpg')
#pic_train <- list.files(path="C:\\Users\\nitin\\OneDrive\\Desktop\\CourseWork\\R Codes_Deep_learning\maize_dataset", pattern="*.png",all.files=F, full.names=F, no.. = F)  
#pic_train
#pic_train <- flow_images_from_directory(directory = "C:\\Users\\nitin\\OneDrive\\Desktop\\CourseWork\\R Codes_Deep_learning\\ml_dataset", generator = image_data_generator(),
                           #target_size = c(256, 256), color_mode = "rgb", classes = NULL,
                           #class_mode = "categorical", batch_size = 32, shuffle = TRUE,
                           #seed = NULL, save_to_dir = NULL, save_prefix = "",
                           #save_format = "png", follow_links = FALSE, subset = NULL,
                           #interpolation = "nearest")
#image_load(path = "C:\\Users\\nitin\\OneDrive\\Desktop\\CourseWork\\R Codes_Deep_learning\\ml_dataset" , grayscale = FALSE, target_size = NULL,
           #interpolation = "nearest")
#pic_train <- as.integer(pic_train)
#pic_train2 <- list.files(path="C:\\Users\\nitin\\OneDrive\\Desktop\\CourseWork\\R Codes_Deep_learning\\sugarbeet", pattern="*.png",all.files=T, full.names=F, no.. = T)
#pic_train2
#str(pics)
#sort <- c(pic_train)
#This imports the files as: files  - chr[1:2] "n04197391_11_0" "n04197391_74_0"
#mypic1 <- list()
#for (i in 1:length(pic_train))
{
 # mypic1[[i]] <- readImage(pic_train[i])
}
#display(mypic1[[1]])
#For installing EBImage Package
if (!requireNamespace("BiocManager", quietly = TRUE))
install.packages("BiocManager")
BiocManager::install("EBImage")
# Set working  Directory
setwd("C:/Users/nitin/OneDrive/Desktop/CourseWork/R Codes_Deep_learning/DataSet")
#Labelling all the pics
pics <- c('m1.png', 'm2.png', 'm3.png', 'm4.png', 'm5.png', 'm6.png', 'm7.png', 'm8.png', 'm9.png', 'm10.png', 'm11.png', 'm12.png', 'm13.png', 'm14.png', 'm15.png', 'm16.png', 'm17.png', 'm18.png', 'm19.png', 'm20.png', 'm21.png', 'm22.png', 'm23.png', 'm24.png', 'm25.png', 'm26.png', 'm27.png', 'm28.png', 'm29.png', 'm30.png', 'm31.png', 'm32.png', 'm33.png', 'm34.png', 'm35.png', 'm36.png', 'm37.png', 'm38.png', 'm39.png', 'm40.png', 'm41.png', 'm42.png', 'm43.png', 'm44.png', 'm45.png', 'm46.png', 'm47.png', 'm48.png', 'm49.png', 'm50.png',                            
          's1.png', 's2.png', 's3.png', 's4.png', 's5.png', 's6.png', 's7.png', 's8.png', 's9.png', 's10.png', 's11.png', 's12.png', 's13.png', 's14.png', 's15.png', 's16.png', 's17.png', 's18.png', 's19.png', 's20.png', 's21.png', 's22.png', 's23.png', 's24.png', 's25.png', 's26.png', 's27.png', 's28.png', 's29.png', 's30.png', 's31.png', 's32.png', 's33.png', 's34.png', 's35.png', 's36.png', 's37.png', 's38.png', 's39.png', 's40.png', 's41.png', 's42.png', 's43.png', 's44.png', 's45.png', 's46.png', 's47.png', 's48.png', 's49.png', 's50.png')
#Creating an empty list
mypic <- list()
#using for loop to read all the images and save it in mypic
for (i in 1:100) {
  mypic[[i]] <- readImage(pics[i])
}
#Printing and checking if the above code is working
print(mypic[[2]])
#displaying the printed pic
display(mypic[[2]])
#Printing the summary, example statistical summaries
summary(mypic[[2]])
#Displaying the histogram of a pic for data analysis
hist(mypic[[2]])
#Printing the structure of all the images
str(mypic)
#resizing all the images to 28 x 28
for (i in 1:100) {
  mypic[[i]] <- resize(mypic[[i]], 28, 28)
}
#again printing the strucure to see if the image is resized
str(mypic)
#Reshaping all the images to array sized i.e., 28 x 28 x 3 = 2352
for (i in 1:100) {
  mypic[[i]] <- array_reshape(mypic[[1]], c(28, 28, 3))
}
#again checking the structure if it has changed after using array_reshape
str(mypic)

#Combining training data on animal dataset using rbind function
train <- NULL
for (i in 51:92) {
  train <- rbind(train, mypic[[i]])
}
str(train)
# For training data on carset using rbind function
#train <- NULL
#for (j in 13:22) {
 # train <- rbind(train, mypic[[i]])
# For testing data
test <- NULL
for (j in 93:100) {
  test <- rbind(test, mypic[[j]])
  
}
str(test)
#test <- rbind(mypic[[]], mypic[[12]], mypic[[23]], mypic[[24]])

train_y <- c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
test_y <- c(0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1)
#One Hot Encoding
trainLabels <- to_categorical(train_y)
testLabels <- to_categorical(test_y)
trainLabels
model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(2352)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')
summary(model)
#compliling the model
model %>%
  compile(loss = 'binary_crossentropy',
          optimizer = optimizer_adadelta(),
          metrics = c('accuracy'))
#fit model
history <- model %>%
  fit(train,
      trainLabels,
      epochs = 30,
      batch_size = 32,
      validation_split = 0.3)
plot(history)
#Evaluation and Prediction
model %>% evaluate(train, trainLabels)
pred <- model %>% predict_classes(train)
pred
#Creating a confusion matrix
table(Predicted = pred, Actual = train_y)
prob <- model %>% predict_proba(train)
prob
cbind(prob, Prected = pred, Actual= train_y)
