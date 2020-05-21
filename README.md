# Scene-Recognition-with-Bag-of-Words-
We will perform scene recognition with three different methods. We will classify scenes into one of 15 categories by training and testing on the 15 scene database (Lazebnik et al. 2006).
the implementation is with three scene recognition schemes:

●	Tiny images representation (get_tiny_images()) and nearest neighbor classifier (nearest_neighbor_classify()).

●	Bag of words representation (build_vocabulary(), get_bags_of_words()) and nearest neighbor classifier.

●	Bag of words representation and linear SVM classifier (svm_classify()).

For detailed project requirements -----> Check "03_CIE552_Proj_SceneRecognitionBoW".
For detailed project implementation ------> Check "Project 3 Report"

To run the project you need to run the following:
1. helpers.py ----> implementation of some helpers ( not core ones ) such as visualization.
2. student.py ----> implementation of the main five functions that do the objective of the project.
3. create_results_webpage.py ----> implementation of the code that make a web page to shoe the result (confusion matrix - Samples of the   results)
4. main.py ----> The driver code that integrate the previous files.

### Data:
To download the data needed for the project ----> Check: https://drive.google.com/drive/folders/14CVYwJG4e18fOKv0UxnAx35YY-Em_9Ts?usp=sharing
For more information about the data, check "03_CIE552_Proj_SceneRecognitionBoW".

