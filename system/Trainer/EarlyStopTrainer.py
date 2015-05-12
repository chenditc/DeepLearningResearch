import numpy
import time
import theano
import sys



class EarlyStopTrainer:
    def __init__(self, model, 
                 dataLoader,
                 config):
        self._startLearningRate = config['startLearningRate']
        self._maxEpoch = config['maxTrainingEpoch']
        self._batch_size = config['batch_size']

        # store model
        self._model = model
        self._dataLoader = dataLoader
        self._train_set_x, self._train_set_y = dataLoader.getTrainingSet() 
        self._valid_set_x, self._valid_set_y = dataLoader.getValidationSet() 
        self._test_set_x, self._test_set_y = dataLoader.getTestSet() 

        self._learningRate = theano.shared(numpy.float(self._startLearningRate))


        # Building training model
        print "#####################################"
        print "Building model: ", self._model.__class__.__name__
        self._model.buildTrainingModel(self._train_set_x, 
                                       self._train_set_y, 
                                       learning_rate = self._learningRate, 
                                       batch_size = self._batch_size, 
                                       parameterToTrain = []) 
        # TODO: turn on properly
#        self._model.setPretrainLayer(layerNumber = 1, batch_size = batch_size, train_set_x = self._train_set_x, learning_rate = self._learningRate)
        print "#####################################"


    def trainModel(self):
        print "#####################################"
        print "Training model"

        ###############
        # TRAIN MODEL #
        ###############
        # early-stopping parameters
        patience = 5000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                              # found

        best_validation_loss = numpy.inf
        self._lastBestEpoch = 0
        start_time = time.clock()

        saveImage = True

        # loop until finish the epoch or explicitly end it by setting variable
        for epoch in range(self._maxEpoch):
            # update the data we use, and then train one epoch
            self._dataLoader.updateTrainingSet()
            self._model.pretrainModel()
            self._model.trainModel()

            # compute zero-one loss on validation set
            this_validation_loss = self._model.getTestError(self._valid_set_x, self._valid_set_y)

            print 'Learning rate: %f, epoch %i, validation error %f %%' % (self._learningRate.get_value(), epoch, this_validation_loss * 100. )

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                self._lastBestEpoch = epoch
                patience +=  patience_increase
                best_validation_loss = this_validation_loss
                # store the model
                # TODO
#                self._model.uploadModel(self._dataLoader, best_validation_loss) 
                print "Learning rate: %f, Get new best validation loss: %f" % (self._learningRate.get_value(), best_validation_loss * 100)

                # Try to save image, if failed, turn this feature off
                if saveImage:
                    try:
                        self._model.saveParameterAsImage("image at %d.png" % epoch)
                    except:
                        saveImage = False
            else:
                # if no new best for too long, lower the learning rate
                if epoch - self._lastBestEpoch > 30:
                    self._learningRate.set_value(self._learningRate.get_value() * 0.95)
                    self._lastBestEpoch = epoch

            sys.stdout.flush()

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
        print 'Total time %f' % (end_time - start_time)

        print "#####################################"
