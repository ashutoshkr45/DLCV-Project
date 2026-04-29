import os
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from multiprocessing import Pool
from tqdm import tqdm

img_dir = 'VOCdevkit/VOC2012/JPEGImages'
cam_dir = 'MCTformer_results/MCTformer_plus/attn-patchrefine-npy'
out_la_dir = 'MCTformer_results/MCTformer_plus/attn-patchrefine-npy-crf_1'
out_ha_dir = 'MCTformer_results/MCTformer_plus/attn-patchrefine-npy-crf_12'

def crf_inference(img, probs, t=10):
    h, w = img.shape[:2]
    n_labels = probs.shape[0]
    
    d = dcrf.DenseCRF2D(w, h, n_labels)
    U = -np.log(probs + 1e-8)
    U = U.reshape((n_labels, -1))
    U = np.ascontiguousarray(U).astype(np.float32)
    img = np.ascontiguousarray(img)
    
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)
    
    Q = d.inference(t)
    return np.array(Q).reshape((n_labels, h, w))

def process_image(img_name):
    try:
        out_la_path = os.path.join(out_la_dir, img_name + '.npy')
        out_ha_path = os.path.join(out_ha_dir, img_name + '.npy')
        
        if os.path.exists(out_la_path) and os.path.exists(out_ha_path):
            return None
            
        img_path = os.path.join(img_dir, img_name + '.jpg')
        cam_path = os.path.join(cam_dir, img_name + '.npy')
        
        img = cv2.imread(img_path)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        cam_dict = np.load(cam_path, allow_pickle=True).item()
        
        # Extract ONLY the classes present in this image
        keys = list(cam_dict.keys())
        cams = np.array(list(cam_dict.values()))
        
        # Low Alpha CRF (alpha=1)
        bg_score_la = np.power(1 - np.max(cams, axis=0, keepdims=True), 1)
        probs_la = np.concatenate((bg_score_la, cams), axis=0)
        probs_la = probs_la / (np.sum(probs_la, axis=0, keepdims=True) + 1e-8)
        crf_la = crf_inference(img, probs_la)
        
        # Save as dictionary mapping class index to CRF probabilities
        dict_la = {0: crf_la[0]}
        for i, k in enumerate(keys): dict_la[k+1] = crf_la[i+1]
        np.save(out_la_path, dict_la)
        
        # High Alpha CRF (alpha=12)
        bg_score_ha = np.power(1 - np.max(cams, axis=0, keepdims=True), 12)
        probs_ha = np.concatenate((bg_score_ha, cams), axis=0)
        probs_ha = probs_ha / (np.sum(probs_ha, axis=0, keepdims=True) + 1e-8)
        crf_ha = crf_inference(img, probs_ha)
        
        dict_ha = {0: crf_ha[0]}
        for i, k in enumerate(keys): dict_ha[k+1] = crf_ha[i+1]
        np.save(out_ha_path, dict_ha)
        
        return None
    except Exception as e:
        return f"Error on {img_name}: {str(e)}"

if __name__ == '__main__':
    os.makedirs(out_la_dir, exist_ok=True)
    os.makedirs(out_ha_dir, exist_ok=True)
    
    img_names = [f[:-4] for f in os.listdir(cam_dir) if f.endswith('.npy')]
    print(f"Applying DenseCRF to {len(img_names)} images...")
    
    with Pool(4) as p: 
        list(tqdm(p.imap(process_image, img_names), total=len(img_names)))
    print("CRF generation complete!")