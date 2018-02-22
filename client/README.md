# fluttermatic

S2 research frontend project for Citeomatic based on flutter.

## Getting Started

Using the frontend requires 4 pieces.

1. Node for the frontend server.
2. Python Flask for the API server.
3. Grobid to parse uploaded PDF files.
4. Citeomatic tensorflow backend service.

To get started you'll need to download Grobid externally and clone the scholar-research repository to get the Citeomatic backend.

Once grobid is downloaded and extracted. I used docker to run it on port 8080. 

`docker run -it --rm -p 8080:8080 lfoppiano/grobid:0.4.1-SNAPSHOT`

After that make sure the tensorflow backend is running. First you'll fetch the model from S3. Ensure both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set in your environment.

`scholar-research/ai2/citeomatic/scripts/fetch_s3_data.sh`

Make sure the MODEL_PATH variable in Procfile points to the model directory you downloaded with the above command otherwise the below command won't work.

Then from this project root run below to run the flask API server and Node server.

`cd scholar-research/ai2/citeomatic/client;` 
`npm install`
`cd ../`
`honcho start`

It will open localhost:5100 but make sure you go to localhost:5000 to hit the API server which serves both Node and the API.

## Structure

