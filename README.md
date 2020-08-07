# Form Completion Rate Model

This project trains an XGBoost tree model to predict the completion rate of a form. (The percentage of users would submit after viewing the form). And expose the model with a RESTFul API built with FastAPI.

Please see the [modelling.ipynb](https://github.com/hhschu/form-completion-rate/blob/master/modelling.ipynb) for the modelling process.

## Train a model

To train and export a model, run

```sh
docker build -f Dockerfile.estimator -t training .
docker run -it --rm --name training-service -v $(pwd)/model:/app/model -v $(pwd)/data:/app/data training --data /app/data/completion_rate.csv --output /app/model/ --steps 20
```

where `$(pwd)/data` is the folder training data is kept and `$(pwd)/model` is the folder where the model will be stored. (You may need to increase the available memory size for the Docker VM.)

There are 4 arguments:

* `--data`: the path of the training data CSV file.
* `--output`: the path of the folder to store the model.
* `--steps`: number of boost round of the XGBoost model.
* `-v`: verbose.

## Run the API server

To start the API server, run

```sh
docker build -f Dockerfile.api -t serving .
docker run -it --rm --name prediction-service -v $(pwd)/model:/app/model -p 80:80 serving
```

where `$(pwd)/model` is the folder where models are kept.

The API has 5 endpoints:

* `GET /`: heartbeat, returns ok.
* `GET /predict`: returns all available model versions.
* `POST /predict`: returns the predicted completion rate of a form.
* `POST /predict/{version}`: returns the predicted completion rate of a form with a specific model version.
* `POST /load/{version}`: load a new model into memory.

An auto-generated API document can also be found at [http://127.0.0.1/docs](http://127.0.0.1/docs).

## Contribution

## Q&A

1. **Analyze the results and document the assumptions and modelling decisions.**

Please see the notebook [modelling.ipynb](https://github.com/hhschu/form-completion-rate/blob/master/modelling.ipynb).

2. **A simple http API to serve online predictions. The API should be able to handle 1,500 POST requests per minute.**

The API can process ~2,400 POST requests per second. Please refer to the Benchmark section for a detailed benchmarking result with `wrk`.

3. **A detailed description of the architecture required to run the pipeline and serve the model through the http API in a cloud environment such as AWS.**

This is a fairly simple service. At the minimum, 1 virtual machine instance can fulfill all the requirements (EC2). We can even use a spot instance to lower the cost, since the training time is short and serving API request is a stateless task. Even if the training time is long, we can still periodically export the training status to be able to utilise a spot instance.

Ideally, both the training data and models should be saved in a blob storage such as S3. This way, they can be easily shared among different instances, and make the instance truly stateless.

To better manage the API service, and to scale up further, we can also put a load balancer in front of the API instances. Or even a manged API service such as AWS API Gateway. With both training and inference processes are containerised, we can also deploy them to a managed Kubernetes cluster, EKS for example, and store the image in a private image repository, such as ECR.

At last, there are managed model serving platforms that help to serve the models. For instance, AWS' SageMaker and GCP's AI Platform. This probably comes with a higher cost but can save some management effort.

In terms of transfroming data, in this example I only use scikit learn and pandas. If the size of data is larger, we can also use Spark to run transformation. Spark also has XGBoost algorithm, so it can also be used to run batch prediction, like what [Uber does](https://eng.uber.com/productionizing-distributed-xgboost/).

Usually, the cloud platform also provides the monitoring and logging service, for example, CloudWatch and Stackdriver. We can use these services to monitor the health of our instances and keeping logs. Or we can use 3rd party services like Datadog or open-source projects like Prometheus.

4. **A description of metrics to capture if the http API had to be instrumented.**

To monitor the API, we should capture:
- the number of requests
- the response time
- the throughput of the endpoint
- the number of successful and failed requests (status code)
- the amount of data received (bytes)
- the source of the request, such as IP address and user-agent
- the payload to check if the data is drifting

## Benchmark

The API performance measured with [wrk](https://github.com/wg/wrk).

```sh
❯ wrk -s wrk.lua -t12 -c400 -d30s http://127.0.0.1/infer
Running 30s test @ http://127.0.0.1/infer
  12 threads and 400 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency   206.51ms   87.86ms 608.59ms   72.17%
    Req/Sec   152.52     43.25   310.00     68.79%
  54788 requests in 30.10s, 8.20MB read
  Socket errors: connect 0, read 322, write 0, timeout 0
Requests/sec:   1820.19
Transfer/sec:    279.07KB
```
