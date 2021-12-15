import segmentation_models_pytorch as smp
import numpy as np

def build_tensorboard_metrics(logs, prefix):
    out = {}
    out[prefix + '_iou_score'] = logs['iou_score']
    out[prefix + '_accuracy'] = logs['accuracy']
    out[prefix + '_fscore'] = logs['fscore']
    out[prefix + '_recall'] = logs['recall']
    out[prefix + '_precision'] = logs['precision']
    out[prefix + '_loss'] = logs['focal_loss']
    return out

def convert_to_onehot(img, mapping):
    nclasses = len(mapping)
    shape = img.shape[:2]+(nclasses,)
    ret = np.zeros(shape, dtype=np.int8)
    for idx, cls in enumerate(mapping):
        ret[:,:,idx] = np.all(img.reshape( (-1,3) ) == mapping[idx], axis=1).reshape(shape[:2])
    return ret

def convert_to_rgb(onehot, mapping):
    val = np.argmax(onehot, axis=-1)
    ret = np.zeros( onehot.shape[:2]+(3,) )
    for m in mapping.keys():
        ret[val == m] = mapping[m]
    return np.uint8(ret)

def get_metrics_to_capture():
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5),
        smp.utils.metrics.Accuracy(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.Precision(threshold=0.5),
    ]
    return metrics

def get_loss_function(name='focal_loss'):
    if name == "dice_loss":
        loss = smp.losses.DiceLoss()
    elif name == "cross_entropy_loss":
        loss = CrossEntropyLossForOneHot()
    else:
        loss = smp.losses.FocalLoss(mode="multilabel")
    
    loss.__name__ = name
    return loss