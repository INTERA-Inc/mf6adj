Supplemental technical information
==================================

Technical detail behind extensions made to mf6adj after the original
publication. Each chapter is self-contained.

* **Instantaneous performance measures** -- a third measure form alongside the
  direct and residual forms, which averages a quantity over the times at which
  it is observed rather than accumulating it, so the result does not depend on
  the time discretization.
* **Storage** -- how the storage terms couple one time step to the next in the
  adjoint, and how the specific-storage and specific-yield contributions are
  selected to match MODFLOW 6.
* **Lake Package** -- bordering the adjoint system with the lake water balance,
  so a sensitivity carries the response of the stage rather than holding it
  fixed.
* **Streamflow Routing Package** -- the same treatment for reach depths,
  including the routing between reaches.

The document is written in LaTeX and lives in ``docs/SuppInfo``. Build it
locally with ``make`` in that directory, which writes ``mf6adjsuppinfo.pdf``.
The copy offered below is built by the documentation workflow and downloaded
into this site with the rendered notebooks.

.. only:: html

   :download:`Download the supplemental technical information (PDF)
   <examples/mf6adjsuppinfo.pdf>`
