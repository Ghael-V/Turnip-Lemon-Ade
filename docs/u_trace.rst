u_trace GPU Performance Tracing
===============================

Mesa has its own GPU performance tracing framework which drivers may
choose to implement. ``gpu.renderstages.*`` producer for
:doc:`Perfetto Tracing <perfetto>` is based on u_trace.

It doesn't require external dependencies and much simpler to use. Though
it provides information only about GPU timings and is harder to analyze
for complex rendering.

u_trace is useful when one needs to quickly identify performance bottleneck,
or to build a tool to analyze the raw performance data.

Drivers which support u_trace:
   - Intel drivers: ANV, Iris
   - Adreno drivers: Freedreno, Turnip
   - AMD drivers: RADV, RadeonSI
   - Arm Mali drivers: Panfrost, PanVK (v10+)
   - Broadcom drivers: V3DV
   - Imagination drivers: PowerVR

Usage
-----

u_trace is controlled by environment variables:

.. envvar:: MESA_GPU_TRACES

   controls whether u_trace is enabled and trace output

   ``print``
      prints in a human readable text format. It should be noted that this
      is mutually exclusive with ``print_json`` and both cannot be enabled
      at the same time.
   ``print_json``
      prints in JSON format, suitable for parsing. Application should
      appropriately finish its rendering in order for trace's json to be
      valid. For the Vulkan API, it is expected to destroy the device,
      for GL it's expected to destroy the context.
   ``perfetto``
      enables Perfetto instrumentation prior to connecting, Perfetto
      traces can be collected without setting this but it may miss some
      events prior to the tracing session being started.
   ``markers``
      enables marker instrumentation, will print trace markers into
      the CS which can then be viewed by dumping the CS from the driver.

         - For Turnip, ``cffdump`` can be used to view the markers in
           the trace.
   ``indirects``
      enables indirect data capture for some of the tracepoints (like
      indirect draw count or indirect dispatch size)

.. envvar:: MESA_GPU_TRACEFILE

   specifies a file where to write the output instead of ``stdout``.
   The output identifies the trace context (e.g. the Vulkan device) that
   wrote it: a final ``ctx=`` field on each CSV row, a ``ctx`` field on
   each JSON frame and a ``(ctx N)`` suffix on the text end-of-frame
   marker.

   If the value contains ``%i``, each context writes to its own file,
   with ``%i`` replaced by its ID; otherwise all contexts share a single
   file. ``print_json`` needs ``%i`` to produce valid JSON with more than
   one context.

   ``%p`` is replaced by the process ID, in every mode. Context IDs are
   only unique within a process, so a variable set for a whole session
   needs this to keep one process from truncating another's trace.

   If the file name ends in ``.gz`` the output is gzip-compressed (needs
   a zlib-enabled build). It is flushed every batch, so a trace stays
   readable up to the last processed submit even if the application
   crashes before exiting.

.. envvar:: *_GPU_TRACEPOINT

   tracepoints can be enabled or disabled using driver specific environment
   variable. Most tracepoints are enabled by default. For instance
   ``TU_GPU_TRACEPOINT=-blit,+render_pass`` will disable the
   ``blit`` tracepoints and enable the ``render_pass`` tracepoints.

   .. list-table::
      :header-rows: 1

      * - Driver
        - Environment Variable
        - Tracepoint Definitions
      * - Freedreno
        - .. envvar:: FD_GPU_TRACEPOINT
        - ``src/gallium/drivers/freedreno/freedreno_tracepoints.py``
      * - Turnip
        - .. envvar:: TU_GPU_TRACEPOINT
        - ``src/freedreno/vulkan/tu_tracepoints.py``
      * - ANV
        - .. envvar:: INTEL_GPU_TRACEPOINT
        - ``src/intel/ds/intel_tracepoints.py``
      * - RADV
        - .. envvar:: RADV_GPU_TRACEPOINT
        - ``src/amd/vulkan/radv_tracepoints.py``
      * - RadeonSI
        - .. envvar:: SI_GPU_TRACEPOINT
        - ``src/gallium/drivers/radeonsi/si_tracepoints.py``
      * - Panfrost
        - .. envvar:: PANFROST_GPU_TRACEPOINT
        - ``src/gallium/drivers/panfrost/panfrost_tracepoints.py``
      * - PowerVR
        - .. envvar:: PVR_GPU_TRACEPOINT
        - ``src/imagination/ds/pvr_tracepoints.py``
