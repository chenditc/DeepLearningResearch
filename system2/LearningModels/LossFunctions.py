
##
# @brief 
# Container of different kinds of loss functions
# 
class LossFunctions:
    @staticmethod
    def negative_log_likelihood(p_y_given_x , true_y):
        import theano.tensor as T
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        #
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] 
        #
        # T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class 
        #
        # LP[T.arange(y.shape[0]),y] is a vector v containing 
        # [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]] 
        # LP[a,b] is picking the value of row a and column b,
        # which represent the probablity function of the true class
        #
        # T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        #
        # The purpose of this function is to raise the probablity of
        # the true y
        return -T.mean(T.log(p_y_given_x)[T.arange(true_y.shape[0]), true_y])
