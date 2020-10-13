import pyspark
import random

sc = pyspark.SparkContext()


### 1st test ###
def firstTest():
    nums = sc.parallelize([1, 2, 3, 4])
    print(nums.map(lambda x: x * x).collect())


### Compute pi ###
NUM_SAMPLES = 100000000


def _inside(p):
    x, y = random.random(), random.random()
    return x * x + y * y < 1


def piSpark():
    count = sc.parallelize(range(0, NUM_SAMPLES)).filter(_inside).count()
    pi = 4 * count / NUM_SAMPLES
    print("Pi is roughly {}".format(pi))


if __name__ == "__main__":
    firstTest()
    piSpark()
