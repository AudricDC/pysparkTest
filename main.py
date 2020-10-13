import os
import pyspark

from pyspark_ml.pyspark_ml import predictWithPipelines

if __name__ == "__main__":
    print(pyspark.__version__)
    predictWithPipelines(input_path=os.path.join(os.getcwd(), "pyspark_ml"))







