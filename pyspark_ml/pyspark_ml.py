import os

from pyspark.sql import SparkSession

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, StringIndexer
from pyspark.sql import Row

from pyspark.ml import Pipeline

spark = SparkSession.builder.getOrCreate()


def _loadDataframe(path):
    rdd = spark.read.text(path).rdd
    rdd_transformed = rdd.map(lambda line: str(line).split(" ")).map(
        lambda words: Row(label=words[0], words=words[1:])).sample(False, 0.15)
    return spark.createDataFrame(rdd_transformed)


def predictVocabulary(input_path):
    # Load train and test data
    train_data = _loadDataframe("20ng-train-all-terms.txt")
    test_data = _loadDataframe("20ng-test-all-terms.txt")

    # Learn the vocabulary of our training data
    vectorizer = CountVectorizer(inputCol="words", outputCol="bag_of_words")
    vectorizer_transformer = vectorizer.fit(train_data)

    # Create bags of words for train and test data
    train_bag_of_words = vectorizer_transformer.transform(train_data)
    test_bag_of_words = vectorizer_transformer.transform(test_data)

    # Convert string labels to floats
    label_indexer = StringIndexer(inputCol="label", outputCol="label_index").setHandleInvalid("skip")
    label_indexer_transformer = label_indexer.fit(train_bag_of_words)
    train_bag_of_words = label_indexer_transformer.transform(train_bag_of_words)
    test_bag_of_words = label_indexer_transformer.transform(test_bag_of_words)

    # Learn multiclass classifier on training data
    classifier = NaiveBayes(
        labelCol="label_index", featuresCol="bag_of_words", predictionCol="label_index_predicted"
    )
    classifier_transformer = classifier.fit(train_bag_of_words)

    # Predict labels on test data
    test_predicted = classifier_transformer.transform(test_bag_of_words)
    test_predicted.limit(10).show()


def predictWithPipelines(input_path):
    train_data = _loadDataframe(os.path.join(input_path, r"20ng-train-all-terms.txt"))
    test_data = _loadDataframe(os.path.join(input_path, r"20ng-test-all-terms.txt"))

    vectorizer = CountVectorizer(inputCol="words", outputCol="bag_of_words")
    label_indexer = StringIndexer(inputCol="label", outputCol="label_index").setHandleInvalid("skip")
    classifier = NaiveBayes(
        labelCol="label_index", featuresCol="bag_of_words", predictionCol="label_index_predicted",
    )
    pipeline = Pipeline(stages=[vectorizer, label_indexer, classifier])
    pipeline_model = pipeline.fit(train_data)

    test_predicted = pipeline_model.transform(test_data)
    test_predicted.limit(10).show()

    # Classifier evaluation
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label_index", predictionCol="label_index_predicted", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(test_predicted)
    print("Accuracy = {:.2f}".format(accuracy))

    input("press ctrl+c to exit")


if __name__ == '__main__':
    predictWithPipelines(input_path=os.getcwd())
