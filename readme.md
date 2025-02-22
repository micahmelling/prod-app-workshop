# Prod App Workshop
This is an example of a production data science application. This repo allows you to 
build machine learning models, evaluate them, and then deploy a selected model via 
an AWS Lambda function managed by Zappa. 

Data documentation can be found in Kaggle: 
https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset?resource=download

The data for the project can be found in csv form in data/raw/. We "mess up" the 
data a bit to make it more "real world"; refer to data/raw/update.py.

To train models, simply run:

```console
$ python3 modeling/train.py
```

This will train the models configured in modeling/config.py; feel free to add 
models. Of note, all scripts are expected to be run from root, and you can install
all requirements with requirements.txt. This project was originally authored in
Python 3.9. 

The script app.py is what gets deployed to Lambda. You will need to change out
the MODEL_ID global for the model ID you want to deploy (see the directory 
structure in modeling/model_results, which is automatically created for you). 

You can then easily deploy the endpoint via Zappa, simply electing the defaults: 
https://github.com/zappa/Zappa

```console
$ zappa init
$ zappa deploy
```

After running the above commands, you will get an endpoint you can send a post
request to; sample_payload.json can be used as a test payload. 

The zappa init command is optional if you opt to use the zappa_settings.json in
the repo. In this case, you will need to fill in the name of the S3 bucket
(you can make something up, and Zappa will create it for you).

We also have to adjust some default settings to deploy a project of this size on
Lambda when using a zip archive to package our project (for simplicity, we are 
taking this route instead of using Docker). In zappa_settings.json in the repo, 
take note of the arguments for slim_handler and ephemeral_storage. Likewise, 
take note of the exclude list, where we can mark packages we do not need for 
the production deployment. Additionally,
the exclude_glob list allows us to remove directories not needed for the 
production app; to note, random_forest_202411302023362858910600 is the ID of the
model I opted to not deploy, so it is excluded.

As a reminder, the proper AWS environment variables need to be set to perform the
deployment. 
```console
$ export AWS_ACCESS_KEY_ID=...
$ export AWS_SECRET_ACCESS_KEY=...
$ export AWS_DEFAULT_REGION=us-west-2
```

Please note that, for a production workflow with money on the line, we would 
likely want to harden our Zappa app (e.g., firewall rules, better logging, 
enforcing least-privilege access, considering Docker for deployment). We opted for
the current deployment for demonstration purposes. 
