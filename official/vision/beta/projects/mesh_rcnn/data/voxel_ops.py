import tensorflow as tf
import scipy.io as sio
import numpy as np

import matplotlib.pyplot as plt


def batch_crop_voxels_within_box(voxels, boxes, Ks, voxel_side_len):
    """
    Batched version of :func:`crop_voxel_within_box`.
    Args:
        voxels (VoxelInstances): store N voxels for an image
        boxes (Tensor): store N boxes corresponding to the masks.
        Ks (Tensor): store N camera matrices
        voxel_side_len (int): the size of the voxel.
    Returns:
        Tensor: A byte tensor of shape (N, voxel_side_len, voxel_side_len, voxel_side_len),
        where N is the number of predicted boxes for this image.
    """
    device = boxes.device
    im_sizes = Ks[:, 1:] * 2.0
    voxels_tensor = torch.stack(voxels.data, 0)
    zranges = torch.stack(
        [voxels_tensor[:, :, 2].min(1)[0], voxels_tensor[:, :, 2].max(1)[0]], dim=1
    )
    cub3D = shape_utils.box2D_to_cuboid3D(zranges, Ks, boxes.clone(), im_sizes)
    txz, tyz = shape_utils.cuboid3D_to_unitbox3D(cub3D)
    x, y, z = voxels_tensor.split(1, dim=2)
    xz = torch.cat([x, z], dim=2)
    yz = torch.cat([y, z], dim=2)
    pxz = txz(xz)
    pyz = tyz(yz)
    cropped_verts = torch.stack([pxz[:, :, 0], pyz[:, :, 0], pxz[:, :, 1]], dim=2)
    results = [
        verts2voxel(cropped_vert, [voxel_side_len] * 3).permute(2, 0, 1)
        for cropped_vert in cropped_verts
    ]

    if len(results) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(results, dim=0).to(device=device)

    
def visualize_voxel(voxel):
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxel, facecolors='red', edgecolor='k')

    plt.show()

def num_vertices(filename):

    file1 = open(filename, 'r')
    lines = file1.readlines()
    num_vertices = 0

    for line in lines:
        if line[0] == 'v':
            num_vertices += 1

    return num_vertices

def num_faces(filename):

    file1 = open(filename, 'r')
    lines = file1.readlines()
    num_faces = 0

    for line in lines:
        if line[0] == 'f':
            num_faces += 1

    return num_faces

def downsample(vox_in, n, use_max=True):
    """
    Downsample a 3-d tensor n times
    Inputs:
      - vox_in (Tensor): HxWxD tensor
      - n (int): number of times to downsample each dimension
      - use_max (bool): use maximum value when downsampling. If set to False
                        the mean value is used.
    Output:
      - vox_out (Tensor): (H/n)x(W/n)x(D/n) tensor
    """
    dimy = vox_in.shape[0] // n
    dimx = vox_in.shape[1] // n
    dimz = vox_in.shape[2] // n
    vox_out = np.zeros((dimy, dimx, dimz))
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                subx = x * n
                suby = y * n
                subz = z * n
                subvox = vox_in[suby : suby + n, subx : subx + n, subz : subz + n]
                if use_max:
                    vox_out[y, x, z] = np.amax(subvox)
                else:
                    vox_out[y, x, z] = np.mean(subvox)
    return vox_out

def verts2voxel(verts, voxel_size):
    def valid_coords(x, y, z, vx_size):
        Hv, Wv, Zv = vx_size
        indx = (x >= 0) * (x < Wv)
        indy = (y >= 0) * (y < Hv)
        indz = (z >= 0) * (z < Zv)
        return indx * indy * indz

    Hv, Wv, Zv = voxel_size
    # create original voxel of size VxVxV
    # orig_voxel = tf.cast(np.zeros((Hv, Wv, Zv)), tf.float32)
    orig_voxel = (np.zeros((Hv, Wv, Zv)))

    x = (verts[:, 0] + 1) * (Wv - 1) / 2
    x = x.astype(np.int32)
    y = (verts[:, 1] + 1) * (Hv - 1) / 2
    y = y.astype(np.int32)
    z = (verts[:, 2] + 1) * (Zv - 1) / 2
    z = z.astype(np.int32)

    keep = valid_coords(x, y, z, voxel_size)
    x = x[keep]
    y = y[keep]
    z = z[keep]

    orig_voxel[(y), (x), (z)] = 1.0

    # align with image coordinate system
    flip_idx = (list(range(Hv)[::-1]))
    orig_voxel = np.take(orig_voxel, indices=flip_idx, axis=0)

    flip_idx = (list(range(Wv)[::-1]))
    orig_voxel = np.take(orig_voxel, indices=flip_idx, axis=1)
    return tf.cast(orig_voxel, tf.float32)


def read_voxel(voxelfile):
    """
    Reads voxel and transforms it in the form of verts
    """
    #Path manager used in original implementation
    with open(voxelfile, "rb") as f:
        voxel = sio.loadmat(f)["voxel"]
    voxel = np.rot90(voxel, k=3, axes=(1, 2))
    verts = np.argwhere(voxel > 0).astype(np.float32, copy=False)

    # centering and normalization
    min_x = np.min(verts[:, 0])
    max_x = np.max(verts[:, 0])
    min_y = np.min(verts[:, 1])
    max_y = np.max(verts[:, 1])
    min_z = np.min(verts[:, 2])
    max_z = np.max(verts[:, 2])
    verts[:, 0] = verts[:, 0] - (max_x + min_x) / 2
    verts[:, 1] = verts[:, 1] - (max_y + min_y) / 2
    verts[:, 2] = verts[:, 2] - (max_z + min_z) / 2
    scale = np.sqrt(np.max(np.sum(verts ** 2, axis=1))) * 2
    verts /= scale
    verts = tf.cast(verts, dtype=tf.float32)

    return verts


def transform_verts(verts, R, t):
    """
    Transforms verts with rotation R and translation t
    Inputs:
        - verts (tensor): of shape (N, 3)
        - R (tensor): of shape (3, 3) or None
        - t (tensor): of shape (3,) or None
    Outputs:
        - rotated_verts (tensor): of shape (N, 3)
    """
    rot_verts = np.array(verts)
    rot_verts = np.transpose(rot_verts)

    if R is not None:
        assert R.ndim == 2
        assert R.shape[0] == 3 and R.shape[1] == 3
        rot_verts = np.matmul(R, rot_verts)
    if t is not None:
        # assert t.ndim == 1
        # assert t.shape[0] == 3
        assert (len(list(t))) == 3
        rot_verts = rot_verts + np.expand_dims(t, axis=1)

    rot_verts = np.transpose(rot_verts) 
    return tf.cast(rot_verts, dtype=tf.float32)

def resize_coordinates(coords, new_w, new_h):
    coords = np.array(coords)
    coords[:, 0] = coords[:, 0] * (new_w * 1.0 / coords.shape[0])
    coords[:, 1] = coords[:, 1] * (new_h * 1.0 / coords.shape[1])
    return tf.cast(coords, dtype=tf.float32)

def horizontal_flip_coordinates(verts):
    verts = np.array(verts)
    verts[:, 0] = -verts[:, 0]
    return tf.cast(verts, dtype=tf.float32)
