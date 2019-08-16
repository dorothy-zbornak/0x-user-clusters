# 0x-user-clusters

Clustering users of 0x Exchange and Forwarder contracts based on call proportions.

## Installation
```bash
git clone https://github.com/dorothy-zbornak/0x-user-clusters
cd 0x-user-clusters/
# Install node packages
yarn -D
# Install python packages.
pip3 install -r ./py/requirements.txt

```

## Usage
There are four steps to follow, when starting from scratch:
1. Pull raw call data.
2. Parse the raw call data.
3. Train a clustering model with data from step 2.
4. Use the clustering model to classify new data generated by step 2.

This repo already comes with a pre-trained model (`/models/model.bin`), which has
been trained on data from Feb 2019 through Aug 2019, so you may choose to skip
step 2.

## Pulling Raw Call Data
This project uses the [`pull-0x-exchange-calls`](https://github.com/dorothy-zbornak/pull-0x-exchange-calls) package to fetch raw call data. As such, you will first need to download your
Google cloud credentials (JSON) file into the root of the project directory as
`/credentials.json`. Any cloud project credentials file *should* work.

To fetch call data starting from a time period, use the `--since` option,
which takes a natural language string. Be patient, this may take a while. For example:

```bash
yarn pull --since "6 months ago"
```

This will create a call dump in `/data/raw-call-data.json`. Do not move this
file, as the other package scripts depend on its location.

## Parsing The Raw Call Data
A lot of the raw data pulled from the previous step is ABI-encoded. Rather than
(slowly) parsing this data on-the-fly every time we want to tweak our cluster
analysis, we do it in a separate script.

Just like the `pull` command, you can slice your data up from a starting time
with the `--since` option, or an ending time with the `--until` option.
Since this step is all run locally, it's probably wise to just pull a large
range of data in step 1 then use this command to split it up into smaller pieces,
as needed.

Don't forget to choose where to save this data with the `--output` option,
otherwise it will just output it to stdout.

```bash
yarn parse --since "1 month ago" --until "1 day ago" --output './data/my-parsed-data.json'
```

## Train a Clustering Model
Now you can train your very own clustering model. Simply pass in the parsed call
data file.

```bash
yarn fit './data/my-parsed-data.json'
```

This should soon display a fancy heatmap of your clustered data. The model that was
trained will be saved to `./models/model.bin` (unless you override it with `--save`).

## Classifying New Data
Now that you have a trained model, you can use it to classify new data that you've
pulled and parsed.

```bash
yarn predict './data/my-other-parsed-data.json'
```

Again, you should see a heatmap of your clustered data. If you want to extract
the labeled data for further processing, you can use the `--output` option.

```bash
yarn predict './data/my-other-parsed-data.json' --output 'clusters.json'
```

## Other Stuff

By default, the `fit` script will create 10 clusters. But you can override this
with the `-c NUM_CLUSTERS` option. But before doing that, you may want to examine
the elbow plot of the clustering inertia to make sure you pick a good number:

```bash
yarn inertia './data/my-parsed-data.json'
```
