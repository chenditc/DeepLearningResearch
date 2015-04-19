import numpy
import time



class EarlyStopTrainer:
    def __init__(self, model, 
                 dataLoader,
                 batch_size = 200,
                 startLearningRate = 0.01, maxEpoch = 10000):
        self._startLearningRate = startLearningRate
        self._maxEpoch = maxEpoch

        # store model
        self._model = model
        self._dataLoader = dataLoader
        self._train_set_x, self._train_set_y = dataLoader.getTrainingSet() 
        self._valid_set_x, self._valid_set_y = dataLoader.getValidationSet() 
        self._test_set_x, self._test_set_y = dataLoader.getTestSet() 

        # Building training model
        print "#####################################"
        print "Building model: ", self._model.__class__.__name__
        self._model.buildTrainingModel(self._train_set_x, self._train_set_y, learning_rate = startLearningRate, batch_size = batch_size) 
        print "#####################################"

    def trainModel(self):
        print "#####################################"
        print "Training model"

        ###############
        # TRAIN MODEL #
        ###############
        # early-stopping parameters
        patience = 500  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                              # found

        best_validation_loss = numpy.inf
        start_time = time.clock()

        # loop until finish the epoch or explicitly end it by setting variable
        for epoch in range(self._maxEpoch):
            # update the data we use, and then train one epoch
            self._dataLoader.updateTrainingSet()
            self._model.trainModel()

            # compute zero-one loss on validation set
            this_validation_loss = self._model.getTestError(self._valid_set_x, self._valid_set_y)

            print 'epoch %i, validation error %f %%' % (epoch, this_validation_loss * 100. )

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                patience +=  patience_increase
                best_validation_loss = this_validation_loss
                # store the model
#                self._model.uploadModel(self._dataLoader, best_validation_loss) 
                print "Get new best validation loss: %f", best_validation_loss * 100

            # if we think the performance is saturated
            # TODO: we should probably lower the training rate here
            if patience < epoch:
                break


        # Finish training phase, test the model and report
        print "############## Training finished ########################"
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
