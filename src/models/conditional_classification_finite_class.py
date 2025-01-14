import torch
import torch.nn as nn
import torch.multiprocessing as mp
from ..utils.helpers import TransformedDataset, LabelMapping
from ..models.projected_stochastic_gradient_descent import SelectorPerceptron

class ConditionalLearnerForFiniteClass(nn.Module):
    def __init__(self, dataset, num_classifier, num_iter, lr_coeff=0.5, train_ratio=0.8, batch_size=32):
        """
        Initialize the conditional learner for finite class classification.

        Parameters:
        dataset (TransformedDataset): The input dataset.
        num_iter (int):               The number of iterations for SGD.
        lr_coeff (float):             The learning rate coefficient.
        train_ratio (float):          The ratio of training samples.
        batch_size (int):             The batch size for SGD.
        """
        super(ConditionalLearnerForFiniteClass, self).__init__()
        self.dataset = dataset
        self.num_classifier = num_classifier
        self.size_sample = len(dataset)
        self.dim_sample = dataset.dim()
        self.num_iter = num_iter
        self.lr_coeff = lr_coeff
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def forward(self, sparse_classifiers):
        """
        Perform conditional learning for finite class classification.


        Parameters:
        sparse_classifiers (torch.Tensor): The list of sparse classifiers.

        Returns:
        selector_list (torch.Tensor): The list of weights for each classifier.
                                      The weight_list is represented as a sparse tensor.
                                      The order of the weight vectors in the list is the same as the following two loops:
                                      for features in feature_combinations:
                                          for samples in sample_combinations:
                                              ...
        """
        
        print("creating selector list ...")
        self.selector_list = torch.zeros([len(sparse_classifiers), self.dim_sample])
        print("start iterating sparse classifiers ...")
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     pool.map(
        #         self.parallel_subroutine, 
        #         [(i, sparse_classifiers[i]) for i in range(len(sparse_classifiers))]
        #     )
        dense_classifiers = sparse_classifiers.to_dense()
        for i in range(len(dense_classifiers)//self.num_classifier):
            classifiers = dense_classifiers[
                self.num_classifier * i : self.num_classifier * (i + 1), 
                :
            ]
            self.parallel_subroutine((i, classifiers))
        
        return self.selector_list
    
    def parallel_subroutine(
            self,
            args
        ):
        i, classifiers = args
        print(f"conditional learner - entering the {i}th iteration ...")
        # print(f"conditional learner - setting transform ...")
        self.dataset.set_transform(LabelMapping(classifiers))
        # print("conditional learner - initializing projected SGD process ...")
        selector_learner = SelectorPerceptron(
            self.dataset, 
            self.num_classifier, 
            self.num_iter, 
            self.lr_coeff, 
            self.train_ratio, 
            self.batch_size
        )
        init_weight = torch.zeros(self.dim_sample, dtype=torch.float32)
        init_weight[0] = 1
        # print("conditional learner - running projected SGD process ...")
        self.selector_list[self.num_classifier * i : self.num_classifier * (i + 1), :], _ = selector_learner(init_weight)