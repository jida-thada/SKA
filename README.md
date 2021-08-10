# SKA
Anonymous repository for <i>Semi-Supervised Knowledge Amalgamation for Sequence Classification (SKA)</i>.
This repository contains the implementation of the first solution for SKA, named <i>Teacher Coordinator (TC)</i>. 

## File listing

+ __main.py__ : Main code for TC training
+ __model.py__ : Supporting models
+ __utils.py__ : Supporting functions
+ __requirements.txt__ : Library requirements

## Instructions on training TC

Prepared folders:

+ __data__ : contains training data
+ __teachers__ : contains teacher models
+ __output__ : directory for training logs and the final student model if saved
    + __plot__ : directory for training loss plots

Note that all datasets and the teacher models used in the paper can be found here: https://drive.google.com/drive/folders/1guPbiUawDvVrkhZXQvMZPS4blGosZiWW?usp=sharing

Run script as:

    python main.py -expname syn_exp1 -t_model ./teachers/exp1_syn_t1.sav ./teachers/exp1_syn_t2.sav \
    -t_numclass 4 4 -t_class 1 2 3 4 3 4 5 6 -s_class 1 2 3 4 5 6 -data ./data/SYN/syn_test.txt
  
<!-- data_label ./data/labeled_data.txt -data_unlabel ./data/unlabeled_data.txt -expname 'test'-->
  
<b>Parameters:</b>

+ __Required:__
  + __-t_model__ : a list of paths of teacher models 
  + __-t_numclass__ : the number of classes corresponding to t_model
  + __-t_class__ : a list of specialized classes of each teacher, concatenated in correspond to t_model , e.g., t1_class: 1 2 3 4 and t2_class: 3 4 5 6, then t_class: 1 2 3 4 3 4 5 6
  + __-s_class__ : a list of comprehensive classes of the student
  + __-data__ : the path of student training data file
  + __-expname__ : experiment name
  <!-- + __-data_label__ the student training data file with labels
  + __-data_unlabel__ the student training data file with no label -->

+ __Student network:__
  + __-lr__ : learning rate, default 0.01
  + __-ep__ : epochs, default 200
  + __-bs__ : batch size, default 8
  + __-layers__ : #layers, default 2
  + __-hiddim__ : #hidden units, default 8

+ __TTL network:__
  + __-lrTTL__ : learning rate, default 0.01
  + __-epTTL__ : epochs, default 500
  + __-bsTTL__ : batch size, default 8
  + __-layersTTL__ : #layers, default 2
  + __-hiddimTTL__ : #hidden units, default 8

+ __Others:__
  + __-inputsize__ : #features, default 1
  + __-seed__ : set seed for reproduction, default 0
  + __-plabel__ : proportion of available labeled data (range = [0,1]), default 0.02
  + __--save__ : boolean parameters, whether to save the student model, default false
  
  
## Case Studies
we have experimented with two Teacher models on the SYN dataset with 2% labeling, as described below. We found that the values computed at each step exactly match the intuition of the method. We describe these case studies below.

In this experiment, we have access to two Teachers. Teacher 1 (T1) specializes in Classes A, B, C, and D while Teacher 2 (T2) specializes in Classes C, D, E, and F. Their expertise overlaps only on Classes C and D.

First, we showcase the Teacher Trust Learner (TTL) overcoming an overconfident teacher via an example from Class E, on which only T2 is an expert. T1 predicts P(y_j | y_j \in Y_k, X) = [0, 0, .99, 0, 0, 0] (confidently-wrong prediction of Class C), while T2 predicts [0, 0, 0, 0, .99, 0] (confidently-correct prediction of Class E). Then, the TTL predicts P(y_j \in Y_k∣X) = [.27,.73], indicating that T2 should be trusted more than T1 (correctly). Finally, rescaling via P(y_j ∣y_j \in Y_k, X)P(y_j \in Y_k∣X) and combining the teachers’ predictions: .27[0, 0, .99, 0, 0,0] + .73[0, 0, 0, 0, .99, 0] = [0, 0, .27, 0, .72, 0], which serves as the surrogate target for the student network. This example clearly shows the TTL overcoming an overconfident teacher (T1) to provide a good surrogate target.

Second, we showcase the TTL effectively preserving accurate predictions for an instance for which both Teachers are experts. On an instance from Class D, T1 predicts P(y_j | y_j \in Y_k, X) = [0, 0, 0, .99, 0, 0] (correct), while T2 also predicts [0, 0, 0, .99, 0, 0] (correct). Then, the TTL predicts P(y_j \in Y_k∣X) = [.5, .5], indicating the teachers should be trusted equally (correct). Finally, rescaling and combining the teachers’ predictions we have the perfect surrogate label: [0, 0, 0, .99, 0, 0].

Third, we showcase an instance from Class A, where Teacher 1 is an expert but Teacher 2 is neither an expert nor confident. T1 predicts P(y_j | y_j \in Y_k, X) = [.99, 0, 0, 0, 0, 0] (correct), while T2 predicts [0, 0, .63, .02, 0, .34] (incorrect). Then, the TTL predicts P(y_j \in Y_k∣X) = [.72, .28], indicating that T1 should be trusted more than T2 (correctly). Finally, the surrogate label after rescaling and combining the teachers’ predictions is [.71, 0, .18, .01,. 10] leading the student model to the right direction.
