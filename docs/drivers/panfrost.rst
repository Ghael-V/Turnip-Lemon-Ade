Panfrost
========

The Panfrost driver stack includes an OpenGL ES implementation for Arm Mali
GPUs based on the Midgard and later architectures, as well as PanVK, a Vulkan
implementation for Bifrost and later.

Which GPUs are conformant to which APIs can be seen from the footnotes in the
table below. Each footnote might only apply to some of the GPUs in the row.

.. note::
   On GPUs where PanVK support is experimental, the driver refuses to load by
   default. Setting PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1 enables it. Experimental
   support comes with no guarantees: it may be broken, may require newer kernel
   driver versions, and may be removed.

The following hardware is currently supported:

.. list-table::
   :header-rows: 1

   * - Models
     - Architecture
     - OpenGL ES
     - OpenGL
     - Vulkan
     - OpenCL
   * - T600, T620, T720
     - Midgard (v4)
     - 2.0
     - 2.1
     -
     - 3.0
   * - T760, T820, T830, T860, T880
     - Midgard (v5)
     - 3.1
     - 3.1
     -
     - 3.0
   * - G71, G72
     - Bifrost (v6)
     - 3.1
     - 3.1
     - 1.3
     - 3.0
   * - G31, G51, G52, G76
     - Bifrost (v7)
     - 3.1 [*]_
     - 3.1
     - 1.3
     - 3.0
   * - G57, G68
     - Valhall (v9)
     - 3.1 [*]_
     - 3.1
     -
     - 3.0
   * - G310, G610
     - Valhall (v10)
     - 3.1 [*]_
     - 3.1
     - 1.4 [*]_
     - 3.0 [*]_
   * - G615, G715
     - Valhall (v11)
     - 3.1
     - 3.1
     - 1.4
     - 3.0
   * - G720
     - 5th Gen (v12)
     - 3.1
     - 3.1
     - 1.4
     - 3.0
   * - G725
     - 5th Gen (v13)
     - 3.1
     - 3.1
     - 1.4
     - 3.0
   * - G1-Pro, G1-Premium, G1-Ultra
     - 5th Gen (v14)
     - 3.1
     - 3.1
     - 1.4
     - 3.0

.. [*] `Mali-G52 OpenGL ES 3.1 submission <https://www.khronos.org/conformance/adopters/conformant-products/opengles#submission_949>`__
.. [*] `Mali-G57 OpenGL ES 3.1 submission <https://www.khronos.org/conformance/adopters/conformant-products/opengles#submission_980>`__
.. [*] `Mali-G610 OpenGL ES 3.1 submission <https://www.khronos.org/conformance/adopters/conformant-products/opengles#submission_1053>`__
.. [*] `Mali-G610 Vulkan 1.4 submission <https://www.khronos.org/conformance/adopters/conformant-products#submission_939>`__
.. [*] `Mali-G310 and Mali-G610 OpenCL 3.0 submission <https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_474>`__

Older Mali GPUs based on the Utgard architecture (Mali-400, Mali-450) are
supported by the :doc:`Lima <lima>` driver, not Panfrost. Lima is also
available as part of Mesa.

Building
--------

Panfrost's OpenGL support is a Gallium driver. Since Mali GPUs are 3D-only and
do not include a display controller, Mesa uses kmsro to support display
controllers paired with Mali GPUs. If your board with a Panfrost supported GPU
has a display controller with mainline Linux support not supported by kmsro,
it's easy to add support, see the commit ``cff7de4bb597e9`` as an example.

Build like ``meson . build/ -Dvulkan-drivers=panfrost
-Dgallium-drivers=panfrost`` for a build directory
``build``.

LLVM is required by Panfrost's compilers at build time.

In case of cross compilation without LLVM,
you can build and install the required tools on the host (with LLVM installed) with
``meson . build-host/ -Dtools=panfrost -Dmesa-clc=enabled -Dinstall-mesa-clc=true
-Dprecomp-compiler=enabled -Dinstall-precomp-compiler=true``
and then use ``-Dmesa-clc=system -Dprecomp-compiler=system`` on the cross compile side.

OpenCL is supported through :doc:`Rusticl <../rusticl>`, using the
``-Dgallium-rusticl=true`` flag, and requires LLVM.

For general information on building Mesa, read :doc:`the install documentation
<../install>`.

Chat
----

Panfrost developers and users hang out on IRC at ``#panfrost`` on OFTC. Note
that registering and authenticating with ``NickServ`` is required to prevent
spam. `Join the chat. <https://webchat.oftc.net/?channels=panfrost>`__

AI policy
---------

Many of the Panfrost developers share the view that important code is best
written by human beings at this point. Writing code is not the bottleneck
of development, and it's critical that we have human beings that deeply
understand the code and are accountable for any changes. The best way of
getting this understanding is by writing the code ourselves.

So we ask developers to refrain from using AI to generate patches. It's OK to
use AI to analyze code, review test logs, sanity-check patches etc., but please
write the code you submit yourself.

We also do not want AI-generated issue reports and other project communication.
This is because reports like that tend to be overly verbose and often
confusing to read. We prefer a short report written by a human over a wall of
text written by a machine. We can always ask for more details when needed.

While we can't conclusively prove violations of this policy, we ask that
contributors do their best to follow it.

Technical details
-----------------

You can read more technical details about Panfrost here:

.. toctree::
   :glob:

   panfrost/*
