import pyspark

sc = pyspark.SparkContext()


def wordCount():
    print("*******")
    lines = sc.textFile("wc_test.txt")
    word_counts = lines.flatMap(lambda line: line.split(' ')) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda count1, count2: count1 + count2) \
        .collect()

    for (word, count) in word_counts:
        print(word, count)


if __name__ == "__main__":
    wordCount()
