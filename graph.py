import matplotlib.pyplot as plt 
import numpy as np

training_error = np.array([
    55260, 56873, 57934, 58418, 58943
])

training_error = 1 - training_error / 60000

test_error = np.array([
    9250, 9509, 9709, 9752, 9819
])

test_error = 1 - test_error / 10000

training_samples = np.array([
    3179, 6376, 12769, 25418, 50995
])

plt.loglog(training_samples, training_error)
plt.loglog(training_samples, test_error)
plt.title('Training and Test Error by Number of Training Samples')
plt.xlabel('Number of Training Samples')
plt.ylabel('Error')
plt.legend(['Training Error', 'Test Error'])
plt.xticks([10 ** 3, 10 ** 4, 10 ** 5])
plt.show()