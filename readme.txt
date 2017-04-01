### perceptron, neural network, and collaborative filtering ###

Source code are in pnncf/src
  - zipMailParse.py, used to parse the mail in .zip format, parameters [zip_path, hos], hos=0, parse ham part, hos=1, parse spam part;
  - netflixParser.py used to parse the netflix data in .zip format;
  - Perceptron.py, source code for perceptron classifier;
  - MLPimpl.py, source code for neural network using sklearn library
  - mbcf.py, source code for collaborative filtering;
  - testP.py, testMLP.py, testCF.py, source code for testing called by .sh code in pnncf/
  
How to compile & run (using shell in command line):
  cd ***/pnncf
  - run the test for perceptron:
    1. modify arguments in runP.sh file,
    2. type: sh runP.sh
  - run the test for neural network:
    1. modify arguments in runMPL.sh,
    2. type: sh runMLP.sh
  - run the test for collaborative filtering:
    1. modify the experimental sample size in runCF.sh file,
    2. type: sh runCF.sh
    
# Report are in pnncf/hw3report.pdf, raw experimental data are available in pnncf/Doc/hw3-experimental-data.ods #

###
References:
http://stackoverflow.com/questions/22646623/how-to-read-text-files-in-a-zipped-folder-in-python
http://stackoverflow.com/questions/36008152/scikit-learn-multilayer-neural-network
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
http://blog.csdn.net/ayw215/article/details/6408694
###