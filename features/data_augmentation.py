"""
Data Augmentation for Time Series
"""
import numpy as np
import pandas as pd

class TimeSeriesAugmenter:
    """
    Create augmented versions of time series data
    """
    
    @staticmethod
    def add_noise(data, noise_level=0.001):
        """Add Gaussian noise to data"""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise
    
    @staticmethod
    def time_warp(data, sigma=0.2):
        """Time warping - stretch or compress time axis"""
        t = np.arange(len(data))
        warp = np.cumsum(np.random.normal(0, sigma, len(data)))
        warp = warp - warp[0]
        warp = warp * (len(data) - 1) / warp[-1]
        warped = np.interp(t, warp, data)
        return warped
    
    @staticmethod
    def magnitude_warp(data, sigma=0.2):
        """Scale magnitude randomly"""
        scale = np.random.normal(1, sigma, data.shape)
        return data * scale
    
    @staticmethod
    def random_crop(data, crop_percent=0.1):
        """Randomly crop a portion of the sequence"""
        crop_size = int(len(data) * crop_percent)
        start = np.random.randint(0, crop_size)
        end = len(data) - np.random.randint(0, crop_size)
        return data[start:end]
    
    @staticmethod
    def window_slice(data, window_percent=0.8):
        """Take a random window slice"""
        window_size = int(len(data) * window_percent)
        start = np.random.randint(0, len(data) - window_size)
        return data[start:start + window_size]
    
    @staticmethod
    def augment_sequence(X, y, augmentation_factor=3):
        """Generate augmented versions of sequences"""
        X_aug = [X]
        y_aug = [y]
        
        for _ in range(augmentation_factor):
            # Randomly choose augmentation method
            method = np.random.choice(['noise', 'warp', 'magnitude'])
            
            if method == 'noise':
                X_new = TimeSeriesAugmenter.add_noise(X)
            elif method == 'warp':
                X_new = TimeSeriesAugmenter.time_warp(X)
            else:
                X_new = TimeSeriesAugmenter.magnitude_warp(X)
            
            X_aug.append(X_new)
            y_aug.append(y)
        
        return np.array(X_aug), np.array(y_aug)
    
    @staticmethod
    def augment_dataset(X, y, augmentation_factor=2):
        """Augment entire dataset"""
        X_aug_list = []
        y_aug_list = []
        
        for i in range(len(X)):
            X_aug, y_aug = TimeSeriesAugmenter.augment_sequence(
                X[i], y[i], augmentation_factor
            )
            X_aug_list.append(X_aug)
            y_aug_list.append(y_aug)
        
        X_aug = np.vstack(X_aug_list)
        y_aug = np.concatenate(y_aug_list)
        
        return X_aug, y_aug
