/*
 * Copyright 2025 Collabora Limited
 * SPDX-License-Identifier: MIT
 */


#ifndef PANFROST_KMOD_H_
#define PANFROST_KMOD_H_

#include "pan_kmod.h"

struct drm_panfrost_submit;

int
panfrost_kmod_submit(struct pan_kmod_dev *pan_kdev,
                     struct drm_panfrost_submit *submit);

#endif /* PANFROST_KMOD_H_ */
