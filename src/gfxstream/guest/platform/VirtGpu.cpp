/*
 * Copyright 2022 Google
 * SPDX-License-Identifier: MIT
 */

#include "VirtGpu.h"

#include <cstdlib>
#include <mutex>

#include "Sync.h"
#include "util/detect_os.h"
#include "util/log.h"
#include "util/os_misc.h"

namespace {

// Guards sDevice: threads can make their first call into the driver at once.
static std::mutex sDeviceMutex;
static VirtGpuDevice* sDevice = nullptr;

// The Apple backend is a stub, so kumquat is the only transport there.
static bool useKumquat() {
#if DETECT_OS_APPLE
    return true;
#else
    return os_get_option("VIRTGPU_KUMQUAT") != nullptr;
#endif
}

}  // namespace

VirtGpuDevice* createPlatformVirtGpuDevice(enum VirtGpuCapset capset, int32_t descriptor) {
    if (useKumquat()) {
        return kumquatCreateVirtGpuDevice(capset, descriptor);
    } else {
        return osCreateVirtGpuDevice(capset, descriptor);
    }
}

VirtGpuDevice* VirtGpuDevice::getInstance(enum VirtGpuCapset capset, int32_t descriptor) {
    std::lock_guard<std::mutex> lock(sDeviceMutex);
    // If kCapsetNone is passed, we return a device that was created with any capset.
    // Otherwise, the created device's capset must match the requested capset.
    // We could support multiple capsets with a map of devices but that case isn't needed
    // currently, and with multiple devices it's unclear how to handle kCapsetNone.
    if (capset != kCapsetNone && sDevice && sDevice->getCapset() != capset) {
        mesa_loge("Requested VirtGpuDevice capset %u, already created capset %u", capset,
                  sDevice->getCapset());
        return nullptr;
    }
    if (!sDevice) {
        sDevice = createPlatformVirtGpuDevice(capset, descriptor);
    }
    return sDevice;
}

void VirtGpuDevice::resetInstance() {
    std::lock_guard<std::mutex> lock(sDeviceMutex);
    if (sDevice) {
        delete sDevice;
        sDevice = nullptr;
    }
}

namespace gfxstream {

SyncHelper* createPlatformSyncHelper() {
    if (useKumquat()) {
        return kumquatCreateSyncHelper();
    } else {
        return osCreateSyncHelper();
    }
}

}  // namespace gfxstream
