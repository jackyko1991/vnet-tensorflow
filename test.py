import numpy as np

a = np.random.rand(2,3,4)

b = np.stack([a,a], axis=-1)
print(b.shape)
print(a.shape)

b = np.append(a[:,:,:,np.newaxis],b, axis=-1)
b = np.append(a[:,:,:,np.newaxis],b, axis=-1)
b = np.append(a[:,:,:,np.newaxis],b, axis=-1)
print(b.shape)