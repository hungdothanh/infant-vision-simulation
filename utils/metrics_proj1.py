import time

def measure_loading_performance(data_loader, total_images=100):
    start_time = time.time()
    images_loaded = 0
    for images in data_loader:
        images_loaded += images.size(0)
        if images_loaded >= total_images:
            break
    end_time = time.time()
    return end_time - start_time