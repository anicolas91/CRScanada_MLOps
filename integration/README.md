## About integration tests

This integration test will use the already pre-composed docker image for the crs-prediction app that we had launched on 02_deployment.

The main reason behind it is because right now we are simulating s3 via Localstack, and since we are not using the paid pro version, permanence is deactivated. This means that if we do docker-compose down and docker-compose up again, all the data saved on the virtual s3 will be gone.

On integration tests for actual s3 buckets, feel free to generate a docker image in this folder or up/down the one already for our current app.

## Running integration tests

We have a bash file that will do everything for us.

To run simply:

1. move to the main project folder and activate your virtual environsment 'CRSenv'.
2. make the bash file executable:
   ```bash
   chmod +x integration/run.sh
   ```
3. run the bash file as:
   ```bash
    ./integration/run.sh
    ```

That's it, have fun.