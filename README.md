# MLOps with Docker and Jenkins: Automating Machine Learning Pipelines

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134908631-2f6c75a5-eef8-45b6-ad2d-2f94cac7a83a.png" />
</p>



# Table of Contents
- [Introduction](#introduction)
- [Setting the strategy](#setting-the-strategy)
- [Defining the Dockerfile](#defining-the-dockerfile)
- [Building the image](#building-the-image)
- [Running a container](#running-a-container)
- [Running commands inside the container](#running-commands-inside-the-container)
- [The testing step](#the-testing-step)
  * [Question for Docker beginners: What is the point of what we have done?](#question-for-docker-beginners-what-is-the-point-of-what-we-have-done)
- [Automating a ML pipeline with Jenkins](#automating-a-ml-pipeline-with-jenkins)
- [Creating Jenkins Jobs](#creating-jenkins-jobs)
  * [The github-to-container job](#the-github-to-container-job)
  * [The preprocessing job](#the-preprocessing-job)
  * [The train job](#the-train-job)
  * [The test job](#the-test-job)
- [References](#references)


# Introduction
The purpose of this repository is to provide an example of how we can use DevOps tools like Docker and Jenkins to automate a Machine Learning Pipeline.
At the end of this post, you will know how to create a pipeline that automatically process raw data, trains a model and returns test accuracy every time we make a change in our repository.

For this task we will use the [Adult census income Dataset](https://www.kaggle.com/uciml/adult-census-income). Target variable is income: a binary variable that indicates if an individual earns more than 50k a year or not.

:ledger: NOTE: As the purpose to this article is to automate a Machine Learning Pipeline, we won't dive into EDA as is out of the scope. If you are curious about that you can check this [Kaggle notebook](https://www.kaggle.com/adro99/from-na-ve-to-xgboost-and-ann-adult-census-income), but is not mandatory in order to understand what is done here.

Ok, so let's start!


# Setting the strategy
Before starting to code, I think is important to understand what is the plan. If you look at the repository you can see three python scripts: is easy to figure out what they do by looking at their names :) . We also have the raw dataset: adult.csv, and a Dockerfile (we will talk about it later). But now I want you to  understand the workflow of this project, and for that the first thing we need to do is to understand what are the inputs and outputs of our Python scripts:


<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134908873-d95fa1d2-e1a3-49d1-9f3f-50d24edadd36.png" />
</p>


As we see in the image, preprocessing.py takes the raw data as input and outputs processed data split into train and test. train.py takes train processed data
as input and outputs the model and a json file where we will store the validation accuracy. test.py takes test processed data and the model as inputs and 
outputs a json file with test accuracy.


With this in mind, now we have a bunch of scripts that we have to run in a certain order, that create a bunch of files that we need to store and access. Furthermore, we want to automate all this process. Nowadays, the best way to manage this issue is using Docker: with this tool you can create an isolated environment with all the dependencies needed to run your code (solving the "if works in my machine" problem!) that makes it all easier. Once we have that, we will be able to automate all the process with Jenkins.

There are 3 concepts on which Docker is based: Containers, Images and Dockerfiles. Is indispensable to understand what they do in order to work with Docker. If you are not familiar with them, here is an intuitive definition:

- Containers: A standard unit of software that packages everything you need to run your application (dependencies, environment variables...)

- Dockerfile: This is a file in which you define everything you want to be inside of a container.

- Image: This is the blueprint needed for running a container. You build an image by executing a Dockerfile.


So, in order to use Docker, you will follow this steps:

1. Define a Dockerfile
2. Build the image
3. Run a container 
4. Run commands inside the container


Let's go step by step:


# Defining the Dockerfile

Here we have to define everything we need to run the pipeline. You can have a look at the Dockerfile in the repository, but if you are not familiar with the syntax it may be overwhelming at first. So what we are going to do here is talk about what we want to specify in it and have a look at the syntax step by step. 

First, we need to specify where we want to run our pipeline. For most of the containerized applications people use to choose a light distribution of Linux, like alpine. However, for our pipeline we will just use an image of jupyter called ```jupyter/scipy-notebook```. In the Dockerfile, we specify the following command:

```dockerfile
FROM jupyter/scipy-notebook
```
Then, we have to install some packages. For this purpose we use the command ```RUN```:

```dockerfile
USER root
RUN apt-get update && apt-get install -y jq
RUN pip install joblib
```

:ledger: NOTE: It may not make much sense now, but we will need ```jq``` in order to access values inside json files, and ```joblib``` in order to serialize and deserialize the model.


Next thing we have to set is the distribution of the files inside the container. We want to build a container that has this structure inside:

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134910233-e26a65f2-7795-44fe-8ca2-d741ab081608.png" />
</p>

:ledger: NOTE: “work” folder is autogenerated by Docker. We are not going to put anything inside.

First we create the folders:

```dockerfile
RUN mkdir model raw_data processed_data results
```

And then we set the directories as environment variables (so we don’t hard code paths all over the code)

```dockerfile
ENV MODEL_DIR=/home/jovyan/model
ENV RAW_DATA_DIR=/home/jovyan/raw_data
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV RESULTS_DIR=/home/jovyan/results
ENV RAW_DATA_FILE=adult.csv
```

Finally, we set the order to copy the scripts and the raw data from our repository. They will be pasted in our container once we create it.

```dockerfile
COPY adult.csv ./raw_data/adult.csv
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py
COPY test.py ./test.py
```


# Building the image

Once we have our Dockerfile specified, we can build the image. The command to do this is:

```bash
sudo -S docker build -t adult-model .
```

We specify the name of the image with ```-t adult-model``` (-t stands for tag) and the path of the Dockerfile with ```.```. Docker automatically picks the file named "Dockerfile".


# Running a container

Now that we have an image (a blueprint for a container), we can build a container!

:ledger: NOTE: we are going to build just one container, but in case you don't know, once we have an image we can build as many containers as we want! This opens up a wide range of possibilities.

The command to run a container is the following:

```bash
sudo -S docker run -d --name model adult-model
```

where -d flag is for detached (runs the container in background). We name it "model" and we specify the image we use (adult-model).

# Running commands inside the container

Now that we have our container running, we can run commands inside it by using ```docker exec```. In this project, we need to execute the scripts in order and then show the results. We can do that by the following commands:

- Run preprocessing.py
```bash
sudo -S docker container exec model python3 preprocessing.py
```
- Run train.py
```bash
sudo -S docker container exec model python3 train.py
```
- Run test.py
```bash
sudo -S docker container exec model python3 test.py
```
- Show validation accuracy and test accuracy
```bash
sudo -S docker container exec model cat /home/jovyan/results/train_metadata.json /home/jovyan/results/test_metadata.json 
```

:ledger: NOTE: If you are curious enough (I guess you are) you will want to know what each script actually does. Don't worry, if you are familiar with basic Machine Learning tools (here I basically use Pandas and SKlearn libraries), you can open the scripts and have a look at the code. It's not a big deal and most of the lines are  commented. If you want a deep understanding or you are looking for more complex models than the one shown here, you can take a look at [this notebook](https://www.kaggle.com/adro99/from-na-ve-to-xgboost-and-ann-adult-census-income). 


# The testing step

When building pipelines is common to have a step dedicated to test if the application is well built and good enough to be deployed into production. In this proyect, we will use a conditional statement that tests if the validation accuracy is higher than a threshold. If it is, the model is deployed. If not, the process stops. The code for doing this is the following:

```bash
val_acc=$(sudo -S docker container exec model  jq .validation_acc /home/jovyan/results/train_metadata.json)
threshold=0.8

if echo "$threshold > $val_acc" | bc -l | grep -q 1
then
	echo 'validation accuracy is lower than the threshold, process stopped'
else
   echo 'validation accuracy is higher than the threshold'
   sudo -S docker container exec model python3 test.py
   sudo -S docker container exec model cat /home/jovyan/results/train_metadata.json /home/jovyan/results/test_metadata.json 
fi
```

As you can see, first we set the two variables we want to compare (validation accuray and the threshold) and then we pass them through a condional statement. If the validation accuracy is higher than the threshold, we will execute the model for the test data and then we will show both test and validation results. If not, the process will stop.

And there we have it! our model is fully containerized and we can run all the steps in our pipeline!


## Question for Docker beginners: What is the point of what we have done?

If you are not familiar with Docker, now you might be asking: Ok, This is all good stuff, but at the end I just have my model and my predictions. I can also get them by runnig my python code and with no need to learn Docker so, what's the point of all this?

I'm glad you asked :sunglasses: .

First, having your Machine Learning models in a Docker container is really useful in order to deploy that model into a production environment. As an example, How many times have you seen code on a tutorial or in a repository that you have tried to replicate, and when running the same code in your machine your screen has filled with red? If we don't like to pass through that, imagine what our customers might feel. With Docker containers, this problem is solved. 

Another reason why Docker is really useful is probably the same reason why you are reading this: to help automating an entire pipeline.

So, without any further ado, let's get straight to the point!


# Automating a ML pipeline with Jenkins

For this step we will use Jenkins, a widely famous open source automation server that provides an endless list of plugins to support building, deploying and automating any project.

For this time, we will build the steps of the pipeline using a tool called jobs. Each job will be a step in our pipeline.

:ledger: NOTE: To keep things running smoothly, you will probably need to configure a few things:
- It is probable that you will experience some problems trying to connect Jenkins with Github if you use Jenkins in your localhost. If that is your case, consider creating a secure URL to your localhost. Best tool I have found to do so is [ngrok](https://ngrok.com/).
- As Jenkins uses it's own user (called jenkins), you may need to give it permissions to execute commands without password. You can do this by opening sudores file with ```sudo visudo /etc/sudoers``` and pasting ```jenkins ALL=(ALL) NOPASSWD: ALL```. 


That being said, let's see what is the plan. We will create 4 jobs:

1. The "github-to-container" job: In this job we will "connect" Jenkins with Github in a way that the job will be triggered everytime we do a commit in
Github. We will also build the Docker image and run a container.
2. The "preprocessing" job: In this step we will execute the preprocessing.py script. This Job will be triggered by the "github-to-container" job.
3. The "train" job: In this job we will execute the train.py script. This Job will be triggered by the "preprocessing" job.
4. The "test" job: In this job we will pass the validation score through our conditional statement. If it is higher than the threshold, we will execute the test.py script and show the metadata (the validation and test accuracy). If the validation score is lower than the threshold, the process will stop and no metadata will be provided.


Once we know what to do, let's go for it!

# Creating Jenkins Jobs

## The github-to-container job

For the github-to-container job, first we need to create a "connection" between github and Jenkins. This is done using webhooks. To create the Webhook, go to your repository in Github, choose settings and select webhooks. Select add webhook. In the Payload URL, pass the URL where you run Jenkins and add "//github-webhook/". For content type choose "application/json". For the answer of "Which events would you like to trigger this webhook?", choose "Just the push event". At the bottom select Active. Select add webhook.

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/135043547-060ce04c-8be2-4d4a-b3ce-021ca23fda35.png" />
</p>


Then, you will need to create a credential in Jenkins in order to access Github. In Jenkins, go to Manage Jenkins

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134875896-833713f0-3329-4c79-baff-a931c0a17c91.png" />
</p>

<br/><br/>

Then select Manage Credentials,
<br/><br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134875914-92a382d0-cafe-4540-b333-121702ee7443.png" />
</p>
<br/><br/>

In "Stores scoped to Jenkins" select Jenkins
<br/><br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134875946-523f440d-e160-4549-a09d-ef772fcc22cf.png" />
</p>
<br/><br/>

Then select "Global credentials (unrestricted)"
<br/><br/>


<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134875953-50ffa506-3209-4770-b936-6d4470eae40b.png" />
</p>
<br/><br/>

and Add credentials
<br/><br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134875976-2a3a3c02-091a-47b9-b976-da038db2fe88.png" />
</p>


Here, for Scope select "Global (Jenkins, nodes, items, all child items, etc)", for username and password write your Github username and password. 
You can leave ID empty as it will be autogenerated. You can also add a description. Finally, click OK.

Now let's build the first job!

In Jenkins, go to New Item,


<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134888795-1c88d767-6e24-4836-9bdf-d9fd05a31c00.png" />
</p>

then give it a name and choose Freestyle project.

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134888825-89517608-f3bd-4eb0-b270-bc2bf81c0e44.png" />
</p>

Next step is to set the configuration. For this step, in Source Code Management choose Git, and then paste the URL of your repository and your Github credentials.



<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/135044643-c69325c3-5c19-491c-b84d-3f5fdc25467f.png" />
</p>


Then, in Build Triggers select "GitHub hook trigger for GITScm polling". Finally in the build section choose Add build step, then Execute shell, and then
write the code to build the image and run the container (we have already talked about it):

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134888860-47a2623a-21c2-4e53-8d6e-9b001e59d6bd.png" />
</p>


Choose save.

<br/><br/>

## The preprocessing job

For the "preprocessing" job, in Source Code Management leave it as None. In Build Triggers, select "Build after other projects are built". Then in Projects to watch enter the name of the first job and select "Trigger only if build is stable".


<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134888882-afc3dc9e-de8a-4896-9bc8-14970f126b36.png" />
</p>



In Build, choose Add build step, then execute shell, and write the code to run preprocessing.py:

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134888896-53df7a8f-462e-47e6-84ee-292350ba008f.png" />
</p>


<br/><br/>

## The train job


The "train" job has the same scheme as the "preprocessing" job, but with a few differences. As you might guess, you will need to write the name of the second job in the Build triggers section:


<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134888904-80bc6da2-1927-499c-bdc7-73eb2b6816d4.png" />
</p>

And in the Build section write the code for running train.py.


<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134888934-212dbf62-2aa9-4df1-8873-1b0b199c6811.png" />
</p>

<br/><br/>

## The test job

For the "test" job, select the "train" job for the Build Triggers section, 

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134888994-a656f340-0b5f-4ce8-9f19-95ac25fd4305.png" />
</p>


and in the Build section write the following code:

```bash
val_acc=$(sudo -S docker container exec model  jq .validation_acc /home/jovyan/results/train_metadata.json)
threshold=0.8

if echo "$threshold > $val_acc" | bc -l | grep -q 1
then
	echo 'validation accuracy is lower than the threshold, process stopped'
else
   echo 'validation accuracy is higher than the threshold'
   sudo -S docker container exec model python3 test.py
   sudo -S docker container exec model cat /home/jovyan/results/train_metadata.json /home/jovyan/results/test_metadata.json 
fi    

sudo -S docker rm -f model
```

I have written it just in case you want to copy paste, but in Jenkins it should look like this:

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134889010-01216ff8-f908-4337-8020-d50b7df4b259.png" />
</p>

Click save and there we have it! Our pipeline is now fully automated!

You can now play with it: try making a commit in Github and see how every step goes automatically. At the end, if the model validation accuracy is higher than the threshold, the model will compute the test accuracy and give back the results. 

:ledger: NOTE: In order to see the output of each step, select the step, click on the first number in the Build section at the bottom left and select Console Output. For the last step, you should see the validation and test accuracy.

I hope you have learned a lot! Thanks for reading!


# References

[Docker for Machine Learning – Part III](https://mlinproduction.com/docker-for-ml-part-3/)

[From DevOps to MLOPS: Integrate Machine Learning Models using Jenkins and Docker](https://towardsdatascience.com/from-devops-to-mlops-integrate-machine-learning-models-using-jenkins-and-docker-79034dbedf1)












