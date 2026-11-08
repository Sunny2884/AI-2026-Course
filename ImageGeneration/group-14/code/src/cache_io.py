from safetensors import safe_open
from safetensors.torch import load_file
from model import FEATURE_SCHEMA,validate_features

def validate_cached_row(row):
    if row['feature_schema']!=FEATURE_SCHEMA: raise ValueError('v4 manifest required')
    with safe_open(row['clip_feature_path'],framework='pt') as f:
        m=f.metadata()
        if m['source_path']!=row['source_path'] or m['fingerprint']!=row['clip_origin_fingerprint']:
            raise ValueError('Cached CLIP source mismatch')
        if tuple(f.get_slice('clip').get_shape())!=(257,1280): raise ValueError('Invalid CLIP shape')

def read_features(row):
    validate_cached_row(row)
    x=load_file(row['clip_feature_path']);validate_features(x)
    return {'clip':x['clip'].float()}
