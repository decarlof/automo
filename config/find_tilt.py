#!/bin/env python3
import sys
import os
import argparse

import dxchange
from skimage.io import imread
from skimage.feature import register_translation as sk_register
from skimage.transform import AffineTransform, warp, FundamentalMatrixTransform
import numpy as np

from automo.register_translation import register_translation
import automo.util as util


def alignment_pass(img, img_180):
    upsample = 200
    # Register the translation correction
    trans = register_translation(img, img_180, upsample_factor=upsample)
    # trans = trans[0]
    # Register the rotation correction
    lp_center = (img.shape[0] / 2, img.shape[1] / 2)
    img = util.realign_image(img, shift=-np.array(trans))

    img_lp = logpolar_fancy(img, *lp_center)
    img_180_lp = logpolar_fancy(img_180, *lp_center)
    result = sk_register(img_lp, img_180_lp, upsample_factor=upsample)
    scale_rot = result[0]
    angle = np.degrees(scale_rot[1] / img_lp.shape[1] * 2 * np.pi)
    return angle, trans


def _transformation_matrix(r=0, tx=0, ty=0, sx=1, sy=1):
    # Prepare the matrix
    ihat = [sx * np.cos(r), sx * np.sin(r), 0]
    jhat = [-sy * np.sin(r), sy * np.cos(r), 0]
    khat = [tx, ty, 1]
    # Make the eigenvectors into column vectored matrix
    new_transform = np.array([ihat, jhat, khat]).swapaxes(0, 1)
    return new_transform


def transform_image(img, rotation=0, translation=(0, 0)):
    """Take a set of transformations and apply them to the image.

    Rotations occur around the center of the image, rather than the
    (0, 0).

    Parameters
    ----------
    translation : 2-tuple, optional
      Translation parameters in (vert, horiz) order.
    rotation : float, optional
      Rotation in degrees.
    scale : 2-tuple, optional
      Scaling parameters in (vert, horiz) order.

    """
    rot_center = (img.shape[1] / 2, img.shape[0] / 2)
    xy_trans = (translation[1], translation[0])
    # move center to [0, 0]
    M0 = _transformation_matrix(tx=-rot_center[0], ty=-rot_center[1])
    # rotate about [0, 0] and translate
    M1 = _transformation_matrix(r=np.radians(rotation), tx=xy_trans[0], ty=xy_trans[1])
    # move center back
    M2 = _transformation_matrix(tx=rot_center[0], ty=rot_center[1])
    M = M2.dot(M1).dot(M0)
    tr = FundamentalMatrixTransform(M)
    out = warp(img, tr, mode='wrap')
    return out


def image_corrections(img, img_180, passes=15):

    img = np.flip(img, 1)
    cume_angle = 0
    cume_trans = np.array([0, 0], dtype=float)
    for pass_ in range(passes):
        # Prepare the inter-translated images
        working_img = transform_image(img, translation=cume_trans, rotation=cume_angle)
        # Calculate a new transformation
        angle, trans = alignment_pass(working_img, img_180)
        # Save the cumulative transformations
        cume_angle += angle
        cume_trans += np.array(trans)
    # Convert translations to (x, y)
    cume_trans = (-cume_trans[1], cume_trans[0])
    return cume_angle, cume_trans


_transforms = {}


def _get_transform(i_0, j_0, i_n, j_n, p_n, t_n, p_s, t_s):
    transform = _transforms.get((i_0, j_0, i_n, j_n, p_n, t_n))
    if transform == None:
        i_k = []
        j_k = []
        p_k = []
        t_k = []
        # get mapping relation between log-polar coordinates and cartesian coordinates
        for p in range(0, p_n):
            p_exp = np.exp(p * p_s)
            for t in range(0, t_n):
                t_rad = t * t_s
                i = int(i_0 + p_exp * np.sin(t_rad))
                j = int(j_0 + p_exp * np.cos(t_rad))
                if 0 <= i < i_n and 0 <= j < j_n:
                    i_k.append(i)
                    j_k.append(j)
                    p_k.append(p)
                    t_k.append(t)
        transform = ((np.array(p_k), np.array(t_k)), (np.array(i_k), np.array(j_k)))
        _transforms[i_0, j_0, i_n, j_n, p_n, t_n] = transform
    return transform


def logpolar_fancy(image, i_0, j_0, p_n=None, t_n=None):
    (i_n, j_n) = image.shape[:2]

    i_c = max(i_0, i_n - i_0)
    j_c = max(j_0, j_n - j_0)
    d_c = (i_c ** 2 + j_c ** 2) ** 0.5

    # how many pixels along axis of radial distance
    if p_n == None:
        p_n = int(np.ceil(d_c))

    # how many pixels along axis of rotation angle
    if t_n == None:
        t_n = j_n

    # step size
    p_s = np.log(d_c) / p_n
    t_s = 2.0 * np.pi / t_n

    (pt, ij) = _get_transform(i_0, j_0, i_n, j_n, p_n, t_n, p_s, t_s)

    transformed = np.random.normal(0.5, 0.2, (p_n, t_n) + image.shape[2:])
    # transformed = np.ones((p_n, t_n) + image.shape[2:], dtype=image.dtype)

    transformed[pt] = image[ij]
    return transformed


def main(arg):

    # Prepare arguments
    parser = argparse.ArgumentParser(
        description='Compare two images and get rotation/translation offsets.')
    parser.add_argument('original_image', help='The original image file', default='auto')
    parser.add_argument('flipped_image', help='Image of the specimen after 180 degrees stage rotation.', default='auto')
    parser.add_argument('--passes', '-p', help='How many iterations to run.',
                        default=15, type=int)
    args = parser.parse_args()

    proj_0_fname = args.original_image
    proj_180_fname = args.flipped_image
    if proj_0_fname == 'auto':
        proj_0_fname = os.path.join('preview', 'proj_norm_00000.tiff')
    if proj_180_fname == 'auto':
        proj_180_fname = os.path.join('preview', 'proj_norm_00001.tiff')

    proj_0 = dxchange.read_tiff(proj_0_fname)
    proj_180 = dxchange.read_tiff(proj_180_fname)
    proj_0 = np.squeeze(proj_0)
    proj_180 = np.squeeze(proj_180)

    proj_0 = util.equalize_histogram(proj_0, proj_0.min(), proj_0.max(), n_bin=1000)
    proj_180 = util.equalize_histogram(proj_180, proj_180.min(), proj_180.max(), n_bin=1000)

    # Perform the correction calculation
    rot, trans = image_corrections(proj_0, proj_180, passes=args.passes)
    rot = -(rot / 2.)
    # Display the result
    msg = "Angle: {:.2f}, transX: {:.2f}px, transY: {:.2f}px".format(rot, trans[0], trans[1])
    print(msg)

    f = open('tilt.txt', 'w')
    f.write(str(rot))
    f.close()

if __name__ == "__main__":
    main(sys.argv[1:])
