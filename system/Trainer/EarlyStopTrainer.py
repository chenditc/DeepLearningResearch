import numpy
import time



class EarlyStopTrainer:
    def __init__(self, model, 
                 dataLoader,
                 startLearningRate = 0.2, maxEpoch = 10000):
        self._startLearningRate = startLearningRate
        self._maxEpoch = maxEpoch

        # store model
        self._model = model

        self._train_set_x, self._train_set_y = dataLoader.getTrainingSet() 
        self._valid_set_x, self._valid_set_y = dataLoader.getValidationSet() 
        test_set_x, test_set_y = dataLoader.getTestSet() 

        # Building training model
        print "#####################################"
        print "Building model: ", self._model.__class__.__name__
        self._model.buildTrainingModel(self._train_set_x, self._train_set_y) 
        print "#####################################"

    def trainModel(self):
        print "#####################################"
        print "Training model"

        ###############
        # TRAIN MODEL #
        ###############
        # early-stopping parameters
        patience = 50  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                                      # found
        improvement_threshold = 0.95  # a relative improvement of this much is
                                      # considered significant
        # compute number of minibatches for training, validation and testing

        best_validation_loss = numpy.inf
        start_time = time.clock()

        # loop until finish the epoch or explicitly end it by setting variable
        for epoch in range(self._maxEpoch):
            # Call trainModel(trainingSet, validationSet) to train the one epoch of the model
            self._model.trainModel()

            # compute zero-one loss on validation set
            this_validation_loss = self._model.getTestError(self._valid_set_x, self._valid_set_y)

            print(
                'epoch %i, validation error %f %%' %
                (
                    epoch,
                    this_validation_loss * 100.
                )
            )

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                   improvement_threshold:
                    patience += epoch * patience_increase

                best_validation_loss = this_validation_loss

                print "Get new best validation loss: %f", best_validation_loss * 100

            if patience < epoch:
                break

        test_score = self._model.getTestError(self._test_set_x, self._test_set_y)
        end_time = time.clock()
        print(
            (
                'Optimization complete with best validation score of %f %%,'
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., test_score * 100.)
        )
        print 'The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time))

        print "#####################################"
