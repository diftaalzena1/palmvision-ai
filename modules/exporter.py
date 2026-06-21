# modules/exporter.py
import io
import zipfile
import pandas as pd
from PIL import Image
from datetime import datetime

def export_to_zip(results_list):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for res in results_list:
            img_bytes = io.BytesIO()
            Image.fromarray(res['annotated']).save(img_bytes, format='PNG')
            zipf.writestr(f"{res['filename']}_annotated.png", img_bytes.getvalue())
            if res['coords']:
                df_coords = pd.DataFrame(res['coords'], columns=['x', 'y'])
                zipf.writestr(f"{res['filename']}_coords.csv", df_coords.to_csv(index=False))
    zip_buffer.seek(0)
    return zip_buffer

def export_to_excel(results_list):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # ── Tambahkan metadata PT, Kebun, Afdeling, Blok, Luas, Potential Spots ──
        summary_data = [{
            'Image': r['filename'],
            'PT': r['pt'],
            'Kebun': r['kebun'],
            'Afdeling': r['afdeling'],
            'Nama Blok': r['nama_blok'],
            'Luas (ha)': r['luas_blok_ha'],
            'Trees': r['total_trees'],
            'Density (trees/ha)': f"{r['density']:.1f}",
            'Potential Spots': len(r.get('rec_coords', [])),
        } for r in results_list]
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        for r in results_list:
            if r['coords']:
                df_coords = pd.DataFrame(r['coords'], columns=['x', 'y'])
                sheet_name = r['filename'][:31]
                df_coords.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output