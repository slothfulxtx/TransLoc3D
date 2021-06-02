import torch
import MinkowskiEngine as ME


def create_collate_fn(dataset, quantization_size=None, create_mask=False):

    def collate_fn(batch):
        meta = {}
        data = {}

        for m, d in batch:
            for k, v in m.items():
                if k not in meta:
                    meta[k] = []
                meta[k].append(v)

            for k, v in d.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)

        for k, v in data.items():
            data[k] = torch.stack(v, 0)

        if dataset.batch_transform is not None:
            # Apply the same transformation on all dataset elements
            data = dataset.batch_transform(data)

        if create_mask:
            positives_mask = [[dataset.catalog[label]['positives'][e]
                               for e in meta['idx']] for label in meta['idx']]
            negatives_mask = [[dataset.catalog[label]['negatives'][e]
                               for e in meta['idx']] for label in meta['idx']]

            positives_mask = torch.tensor(positives_mask)
            negatives_mask = torch.tensor(negatives_mask)

            data['pos_mask'] = positives_mask
            data['neg_mask'] = negatives_mask

        if quantization_size is not None:
            for k in list(data.keys()):
                if not k.endswith('pcd'):
                    continue
                coords = [ME.utils.sparse_quantize(coords=e, quantization_size=quantization_size)
                          for e in data[k]]
                # coords = ME.utils.batched_coordinates(coords)
                # feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
                feats = [torch.ones((coord.shape[0], 1),
                                    dtype=torch.float32) for coord in coords]
                data[k+'_coords'] = coords
                data[k+'_feats'] = feats
                pcds = [e for e in data[k]]
                del data[k]
                data[k] = pcds

        return meta, data

    return collate_fn
