# modules/recommendation.py
import numpy as np
from scipy.spatial import cKDTree

def find_empty_spots(coords, img_shape):
    if not coords or len(coords) < 10:
        return []

    coords_np = np.array(coords)
    h, w = img_shape
    
    # 1. Hitung jarak tanam rata-rata
    tree_original = cKDTree(coords_np)
    distances, _ = tree_original.query(coords_np, k=2)
    avg_dist = np.median(distances[:, 1])
    
    # 2. Buat Grid Kandidat
    step = int(avg_dist * 0.2) 
    margin = int(avg_dist * 0.8)
    grid_x, grid_y = np.mgrid[margin:w-margin:step, margin:h-margin:step]
    candidates = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # 3. Filter Jarak: Harus cukup jauh dari pohon asli (tidak tumpang tindih)
    dist_to_green, _ = tree_original.query(candidates)
    
    # Kita buat batas: minimal 0.8x jarak tanam agar tidak tabrakan
    # Maksimal 4.0x agar menjangkau area luas
    potential_spots = candidates[
        (dist_to_green >= avg_dist * 0.8) & (dist_to_green <= avg_dist * 4.0)
    ]
    
    if len(potential_spots) == 0:
        return []

    # 4. PROSES PENYELARASAN (Poisson Disk Sampling Sederhana)
    # Ini kunci agar titik merah berbaris rapi, bukan bergerombol
    final_recommendations = []
    
    # Acak kandidat agar pengisian tidak terlalu kaku/linear
    np.random.seed(42)
    np.random.shuffle(potential_spots)
    
    # Jarak aman antar titik merah (95% dari jarak tanam asli)
    min_dist_allowed = avg_dist * 0.95
    
    accepted_points = []
    
    for p in potential_spots:
        # Jika sudah ada titik merah yang diterima, cek jaraknya
        if accepted_points:
            tree_red = cKDTree(accepted_points)
            dist_to_red, _ = tree_red.query(p)
            if dist_to_red < min_dist_allowed:
                continue
        
        # Tambahan: Cek ulang jarak ke pohon hijau agar benar-benar presisi
        dist_to_green_final, _ = tree_original.query(p)
        if dist_to_green_final < min_dist_allowed:
            continue
            
        accepted_points.append(p)
        final_recommendations.append(p.astype(int))

    return final_recommendations