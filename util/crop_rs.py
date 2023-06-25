import random
import os
import pyproj
import rasterio
from shapely.geometry.polygon import Polygon
from rasterio.mask import mask


def crop(id2name, store_prefix, ref_path, train=True, num_per_src=200, xsize=512, ysize=512):
    ref = rasterio.open(ref_path)
    
    source_dir = os.path.join(store_prefix, "source")
    ref_dir = os.path.join(store_prefix, "ref")
    
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
        
    for source_id in id2name.keys():
        src = rasterio.open(id2name[source_id])
        
        p = pyproj.CRS(src.crs)
        p_to = pyproj.CRS.from_user_input(ref.crs)
        transformer_to_ref = pyproj.Transformer.from_crs(p, p_to, always_xy=True)
                    
        msk = src.read_masks()
        msk = (msk[0] & msk[1] & msk[2])

        src_transform = src.transform
        cnt = 0
        cnt_offset = num_per_src * source_id

        if train:
            xmin, xmax = src.height // 4, src.height - ysize - 1 # row
        else:
            xmin, xmax = 0, src.height // 4 - ysize - 1 # row
        
        # xmin, xmax = 0, src.height - ysize - 1 # row
        ymin, ymax = 0, src.width - xsize - 1 # col

        while True:
            xoff, yoff = random.randint(xmin, xmax), random.randint(ymin, ymax)
            x1, y1 = rasterio.transform.xy(src_transform, xoff, yoff)
            x2, y2 = rasterio.transform.xy(src_transform, xoff + xsize, yoff + ysize)
            source_name = os.path.join(source_dir, f'{cnt + cnt_offset}.tif')
            ref_name = os.path.join(ref_dir, f'{cnt + cnt_offset}.tif')
            
            # filter nodata
            mask1 = msk[xoff, yoff]
            mask2 = msk[xoff, yoff + ysize]
            mask3 = msk[xoff + xsize, yoff]
            mask4 = msk[xoff + xsize, yoff + ysize]
            
            if mask1 <= 0 or mask2 <= 0 or mask3 <= 0 or mask4 <= 0:
                print("\ninvalid extent, continuing...\n")
                continue
            
            # crop source
            coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            copenhagen_poly = Polygon(coords)
            
            cropped_src, cropped_src_transform = mask(src, shapes=[copenhagen_poly], crop=True, all_touched=True)
            profile = src.profile
            profile.update({
                'height': cropped_src.shape[1],
                'width': cropped_src.shape[2],
                'transform': cropped_src_transform
                })
            with rasterio.open(source_name, 'w', **profile) as dst:
                dst.write(cropped_src)
            
            
            # crop reference
            x1, y1 = transformer_to_ref.transform(x1, y1)
            x2, y2 = transformer_to_ref.transform(x2, y2)
            coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            copenhagen_poly = Polygon(coords)

            cropped_ref, cropped_ref_transform = mask(ref, shapes=[copenhagen_poly], crop=True, all_touched=True)
            profile = ref.profile
            profile.update({
                'height': cropped_ref.shape[1],
                'width': cropped_ref.shape[2],
                'transform': cropped_ref_transform
                })
            with rasterio.open(ref_name, 'w', **profile) as dst:
                dst.write(cropped_ref)

            print(f'\nprocessing {cnt}')
            print(source_name)
            print()
            
            cnt = cnt + 1
            if cnt == num_per_src:
                break
        
   
if __name__ == "__main__":
    
    # HR path
    id2name = {
        0: "/home/DL/data/cuiyc/timelapse/Deep/data/rgb/1.tif",
        1: "/home/DL/data/cuiyc/timelapse/Deep/data/rgb/2.tif",
        2: "/home/DL/data/cuiyc/timelapse/Deep/data/rgb/3.tif",
    }
    
    # reference path
    ref_path = "/home/DL/data/cuiyc/timelapse/Deep/preprocess/ref.tif"
    train_store_prefix = "/data/cyc/dataset/train"
    test_store_prefix = "/data/cyc/dataset/test"
    
    # note: source and reference are under the same coordinate system
    # or you will need `pyproj` or `rasterio` to transfom them to the same
    crop(id2name=id2name, ref_path=ref_path, store_prefix=train_store_prefix, num_per_src=600)
    crop(id2name=id2name, ref_path=ref_path, store_prefix=test_store_prefix, train=False, num_per_src=200)
    
    