from seam_carving import FastSeamCarver
from matplotlib import pyplot as plt
# Initialize with image
carver = FastSeamCarver("Input.jpg")

# Energy Map
energy_map = carver.calculate_energy()
plt.imshow(energy_map,cmap='gray')
plt.show()

# Visualize seams (shortest paths)
seam_vis = carver.visualize_seam(num_seams=100)
plt.imshow(seam_vis)
plt.show()
# Resize by removing seams
resized = carver.resize(new_width=700)

# Save result
from PIL import Image
Image.fromarray(resized).save("Output.jpg")