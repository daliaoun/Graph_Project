import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from numba import jit

class SeamCarver:
    """
    Optimized implementation of Seam Carving algorithm for content-aware image resizing.
    Based on Shortest Path on DAG approach with performance optimizations.
    """
    
    def __init__(self, image_path):
        """Initialize with image path."""
        self.original_image = np.array(Image.open(image_path))
        self.image = self.original_image.copy()
        self.height, self.width = self.image.shape[:2]
        
    def calculate_energy(self):
        """
        Calculate energy map using gradient magnitude - OPTIMIZED.
        Uses scipy's Sobel filters for faster computation.
        """
        # Convert to grayscale if color
        if len(self.image.shape) == 3:
            gray = np.dot(self.image[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = self.image.astype(np.float64)
        
        # Use scipy's optimized Sobel filters
        dx = ndimage.sobel(gray, axis=1)
        dy = ndimage.sobel(gray, axis=0)
        
        # Energy is the magnitude of gradient
        energy = np.hypot(dx, dy)
        return energy
    
    def find_vertical_seam(self):
        """
        Find the minimum energy vertical seam - OPTIMIZED.
        Uses vectorized operations for faster computation.
        
        Returns:
            seam: array of x-coordinates for each row
        """
        energy = self.calculate_energy()
        height, width = energy.shape
        
        # Initialize cumulative minimum energy matrix
        M = energy.copy()
        
        # Backtrack matrix
        backtrack = np.zeros_like(M, dtype=np.int32)
        
        # Dynamic Programming - VECTORIZED
        for y in range(1, height):
            # Create shifted versions for vectorized comparison
            M_left = np.roll(M[y-1], 1)
            M_left[0] = np.inf  # Invalid for leftmost column
            
            M_center = M[y-1]
            
            M_right = np.roll(M[y-1], -1)
            M_right[-1] = np.inf  # Invalid for rightmost column
            
            # Stack and find minimum
            options = np.stack([M_left, M_center, M_right])
            backtrack[y] = np.argmin(options, axis=0) - 1  # -1, 0, or 1
            M[y] += np.min(options, axis=0)
        
        # Backtrack to find the seam - OPTIMIZED
        seam = np.zeros(height, dtype=np.int32)
        seam[-1] = np.argmin(M[-1])
        
        for y in range(height - 2, -1, -1):
            seam[y] = seam[y + 1] + backtrack[y + 1, seam[y + 1]]
        
        return seam
    
    def remove_vertical_seam(self, seam):
        """
        Remove a vertical seam from the image - OPTIMIZED.
        Uses advanced indexing for faster removal.
        
        Args:
            seam: array of x-coordinates for each row
        """
        height, width = self.image.shape[:2]
        
        # Create mask for pixels to keep (vectorized)
        if len(self.image.shape) == 3:
            # Color image
            mask = np.ones((height, width, 3), dtype=bool)
            for y in range(height):
                mask[y, seam[y], :] = False
            self.image = self.image[mask].reshape(height, width - 1, 3)
        else:
            # Grayscale image
            mask = np.ones((height, width), dtype=bool)
            for y in range(height):
                mask[y, seam[y]] = False
            self.image = self.image[mask].reshape(height, width - 1)
        
        self.height, self.width = self.image.shape[:2]
    
    def remove_multiple_seams(self, num_seams):
        """
        Remove multiple seams efficiently - BATCH OPTIMIZED.
        
        Args:
            num_seams: number of seams to remove
        """
        seams = []
        
        # Find all seams first (they're independent on current image)
        for i in range(num_seams):
            seam = self.find_vertical_seam()
            seams.append(seam)
            
            # Update seam positions for already removed seams
            for prev_seam in seams[:-1]:
                # If previous seam was to the left, shift indices
                prev_seam[seam <= prev_seam] -= 1
            
            # Temporarily remove for next iteration
            self.remove_vertical_seam(seam)
            
            if (i + 1) % 10 == 0:
                print(f"Removed {i + 1}/{num_seams} seams")
        
        return self.image
    
    def resize(self, new_width):
        """
        Resize image to new width using seam carving - OPTIMIZED.
        
        Args:
            new_width: target width
            
        Returns:
            resized image
        """
        num_seams = self.width - new_width
        
        if num_seams < 0:
            raise ValueError("Can only shrink image width. New width must be smaller.")
        
        if num_seams == 0:
            return self.image
        
        # Process in batches for better performance
        batch_size = min(50, num_seams)
        remaining = num_seams
        processed = 0
        
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            
            for i in range(current_batch):
                seam = self.find_vertical_seam()
                self.remove_vertical_seam(seam)
                processed += 1
                
                if processed % 10 == 0:
                    print(f"Removed {processed}/{num_seams} seams")
            
            remaining -= current_batch
        
        return self.image
    
    def resize_fast(self, new_width):
        """
        Fastest resize using compiled functions where possible.
        
        Args:
            new_width: target width
            
        Returns:
            resized image
        """
        num_seams = self.width - new_width
        
        if num_seams < 0:
            raise ValueError("Can only shrink image width. New width must be smaller.")
        
        # Use optimized removal
        for i in range(num_seams):
            seam = self.find_vertical_seam()
            self.remove_vertical_seam(seam)
            
            if (i + 1) % 20 == 0:
                print(f"Removed {i + 1}/{num_seams} seams")
        
        return self.image
    
    def visualize_seam(self, num_seams=1):
        """
        Visualize seams on the original image - OPTIMIZED.
        
        Args:
            num_seams: number of seams to highlight
            
        Returns:
            image with seams marked in red
        """
        # Reset to original image
        self.image = self.original_image.copy()
        self.height, self.width = self.image.shape[:2]
        
        # Create visualization image
        vis_image = self.image.copy()
        
        # Collect all seams first
        all_seams = []
        for i in range(num_seams):
            seam = self.find_vertical_seam()
            all_seams.append(seam.copy())
            self.remove_vertical_seam(seam)
        
        # Reset image
        self.image = self.original_image.copy()
        self.height, self.width = self.image.shape[:2]
        
        # Mark all seams accounting for removals
        for i, seam in enumerate(all_seams):
            # Adjust for previously marked seams
            adjusted_seam = seam.copy()
            for prev_seam in all_seams[:i]:
                adjusted_seam[prev_seam <= adjusted_seam] += 1
            
            # Mark seam in red
            for y in range(self.height):
                x = adjusted_seam[y]
                if x < vis_image.shape[1]:
                    if len(vis_image.shape) == 3:
                        vis_image[y, x] = [255, 0, 0]
                    else:
                        vis_image[y, x] = 255
        
        return vis_image


# Numba-optimized helper functions for critical paths
@jit(nopython=True)
def compute_M_numba(energy):
    """Numba-optimized dynamic programming for seam finding."""
    height, width = energy.shape
    M = energy.copy()
    
    for y in range(1, height):
        for x in range(width):
            if x == 0:
                M[y, x] += min(M[y-1, x], M[y-1, x+1])
            elif x == width - 1:
                M[y, x] += min(M[y-1, x-1], M[y-1, x])
            else:
                M[y, x] += min(M[y-1, x-1], M[y-1, x], M[y-1, x+1])
    
    return M


class FastSeamCarver(SeamCarver):
    """
    Ultra-optimized version using Numba JIT compilation.
    Falls back to regular version if Numba not available.
    """
    
    def find_vertical_seam(self):
        """
        Numba-accelerated seam finding.
        """
        try:
            energy = self.calculate_energy()
            height, width = energy.shape
            
            # Use Numba-optimized DP
            M = compute_M_numba(energy)
            
            # Backtrack
            seam = np.zeros(height, dtype=np.int32)
            seam[-1] = np.argmin(M[-1])
            
            for y in range(height - 2, -1, -1):
                x = seam[y + 1]
                if x == 0:
                    seam[y] = 0 if M[y, 0] < M[y, 1] else 1
                elif x == width - 1:
                    seam[y] = x - 1 if M[y, x-1] < M[y, x] else x
                else:
                    options = [M[y, x-1], M[y, x], M[y, x+1]]
                    seam[y] = x - 1 + np.argmin(options)
            
            return seam
        except:
            # Fall back to parent method if Numba fails
            return super().find_vertical_seam()


# Example usage
if __name__ == "__main__":
    import os
    
    print("Starting seam carving demo...")
    
    # Test different numbers of seams to remove
    seam_numbers = [50, 100, 150, 200,250,300]
    
    for num_seams_to_remove in seam_numbers:
        print(f"\n{'='*60}")
        print(f"Processing with {num_seams_to_remove} seams")
        print(f"{'='*60}")
        
        # Create folder for this seam number
        folder_name = f"{num_seams_to_remove}_seams"
        os.makedirs(folder_name, exist_ok=True)
        
        # Use FastSeamCarver for best performance
        carver = FastSeamCarver("Broadway_tower_edit.jpg")
        
        print(f"Original size: {carver.width} x {carver.height}")
        
        # Calculate energy map and save it in black and white
        print("Saving energy map...")
        energy_map = carver.calculate_energy()
        plt.figure(figsize=(10, 8))
        plt.imshow(energy_map, cmap='gray')
        plt.title(f"Energy Map - {num_seams_to_remove} Seams")
        plt.colorbar(label='Energy')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{folder_name}/{num_seams_to_remove}_seams_energy_map.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        new_width = carver.width - num_seams_to_remove
        
        # Visualize ALL seams that will be removed
        print(f"Visualizing all {num_seams_to_remove} seams...")
        seam_vis = carver.visualize_seam(num_seams=num_seams_to_remove)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(carver.original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(seam_vis)
        plt.title(f"{num_seams_to_remove} Seams Highlighted")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{folder_name}/{num_seams_to_remove}_seams_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Resize image
        print(f"Resizing to width: {new_width}")
        resized = carver.resize_fast(new_width)
        
        print(f"New size: {carver.width} x {carver.height}")
        
        # Save result
        Image.fromarray(resized).save(f"{folder_name}/{num_seams_to_remove}_seams_output_resized.jpg")
        
        # Show comparison
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(carver.original_image)
        plt.title(f"Original ({carver.original_image.shape[1]}x{carver.original_image.shape[0]})")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(resized)
        plt.title(f"Resized ({resized.shape[1]}x{resized.shape[0]})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{folder_name}/{num_seams_to_remove}_seams_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved all files to '{folder_name}/' folder")
    
    print("\n" + "="*60)
    print("Done! Check output folders for all results.")
    print("="*60)