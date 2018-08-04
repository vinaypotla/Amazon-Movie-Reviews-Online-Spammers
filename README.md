# Amazon-Movie-Reviews-Online-Spammers
Big data

- The raw dataset can be downloaded from http://snap.stanford.edu/data/web-Movies.html
- Execute in the directory of the movies.txt file:
  python process.py 
- Again in the same directory execute:
  python further_process.py

- The preprocessed dataset would be generated

- JAR file and the compressed project files are provided
- class Name: spammers
- Execute the following command to run the program locally:

  spark-submit --class [className] [jar file] [path to preprocessed dataset] [path to AFINN sentiment datatset] [output directory to store Logistic Regression results]   [output directory to store RandomForest results] [output directory to store evaluation metrics]  

  eg: spark-submit --class spammers jar_file csv_file .\AFINN-111.txt .\o1 .\o2 .\o3 


