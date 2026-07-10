"""
physics/constants.py
----------------------
Default physical/optical parameters for the lithography simulator:
working wavelength and numerical aperture (NA).

WHY THIS MODULE EXISTS IN THE PIPELINE
-----------------------------------------
Through Week 8, every function that needed a wavelength (fraunhofer_pattern,
check_fraunhofer_validity, etc.) took it as a required explicit argument,
with no notion of a project-wide "default" value — reasonable when only
diffraction.py needed it. Starting this week, physics/lens.py needs both
wavelength AND numerical aperture (NA) to define the pupil cutoff in
frequency space, and both Week 10 (ATF/OTF) and Week 11 (aberrations)
will need the same two numbers again. Without a single source of truth,
changing the simulator's working wavelength would mean hunting down every
call site by hand — exactly the kind of duplication the project
conventions forbid.

This module holds only DEFAULTS. Every function elsewhere in the project
that uses wavelength or NA still accepts them as explicit optional
parameters (matching diffraction.py's existing style of never hiding a
physical parameter inside a function body) — constants.py just gives
scripts and notebooks a sensible, named starting point instead of a bare
magic number like 0.5 or 0.365 typed inline.

WHY THESE SPECIFIC VALUES
----------------------------
This is a lithography simulator, not a generic optics demo, so the
defaults are chosen to be representative of real photolithography rather
than arbitrary round numbers:

  WAVELENGTH  = 0.365 µm  — the mercury i-line, one of the classic
                 lithography exposure wavelengths (later nodes moved to
                 KrF at 0.248 µm and ArF at 0.193 µm). i-line is used here
                 as the default because it keeps early-project feature
                 sizes and pupil cutoffs in a range that's easy to sanity
                 check by hand against Week 1-8 numbers, which were not
                 chosen with a specific short wavelength in mind.
  NA_DEFAULT  = 0.5       — a modest, "start of the imaging chapters"
                 numerical aperture. Real production lithography tools run
                 much higher (>0.9, or >1 for immersion systems), but a
                 smaller NA gives a wider, easier-to-visualize pupil
                 cutoff in the frequency-domain plots this week's build
                 goal explicitly asks for, before later weeks push NA
                 higher to study its effect on resolution.

Both are ordinary module-level floats, not a frozen dataclass or enum —
there's exactly two values here, and every consumer already takes them as
plain float arguments, so a heavier structure would be unjustified
complexity for what this module needs to do.

All wavelengths and spatial coordinates: µm
NA: dimensionless (sin of the marginal ray half-angle, per Goodman's
    definition used throughout Ch. 6)
"""

WAVELENGTH: float = 0.365
"""Default working wavelength, µm (mercury i-line)."""

NA_DEFAULT: float = 0.5
"""Default numerical aperture, dimensionless."""