
import numpy

number_random_generator = numpy.random.RandomState(123)
def getRandomNumpyMatrix(row, column):
    matrix = number_random_generator.uniform(
        low=-4 * numpy.sqrt(6. / (row + column)),
        high=4 * numpy.sqrt(6. / (row + column)),
        size=(row, column)
    )
    return matrix
