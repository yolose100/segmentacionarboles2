import matplotlib
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte

import skimage.io



mask1 = skimage.io.imread('../outfile.png' , as_gray=True)
mask2 = skimage.io.imread('../outfileSi.png' , as_gray=True)
threshold_global_otsu = threshold_otsu(mask1)
global_otsu1 = mask1 >= threshold_global_otsu
threshold_global_otsu = threshold_otsu(mask2)
global_otsu2 = mask2 >= threshold_global_otsu
resultado = global_otsu1^ global_otsu2



matplotlib.rcParams['font.size'] = 9



fig, ax = plt.subplots(2, 2, figsize=(8, 5))
ax1, ax2, ax3, ax4 = ax.ravel()

fig.colorbar(ax1.imshow(mask1, cmap=plt.cm.gray),
           ax=ax1, orientation='horizontal')
ax1.set_title('Original 1')
ax1.axis('off')

fig.colorbar(ax2.imshow(mask2, cmap=plt.cm.gray),
           ax=ax2, orientation='horizontal')
ax2.set_title('Original 2 ')
ax2.axis('off')

ax3.imshow(resultado, cmap=plt.cm.gray)
ax3.set_title('resta' % threshold_global_otsu)
ax3.axis('off')

plt.show()