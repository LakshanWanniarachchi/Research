/*
 * Drawing an ECG on a canvas - shared by the live monitor and the report page.
 *
 * One implementation rather than one per page: the live trace and the stored
 * report must look the same, or a reader comparing them is comparing two
 * renderers as much as two recordings.
 */

const ECG = (function () {
  "use strict";

  function fitToDisplay(canvas) {
    // The canvas is sized in CSS but drawn in device pixels. Without this the
    // trace is blurred on a high-DPI screen, which matters when the thing
    // being read is the shape of a QRS complex.
    const ratio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * ratio));
    const h = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    return { w: canvas.width, h: canvas.height, ratio: ratio };
  }

  function grid(ctx, w, h, seconds) {
    // ECG paper: a light line every 0.2 s, a darker one every second.
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 1;
    for (let s = 0; s <= seconds; s += 0.2) {
      const x = (s / seconds) * w;
      ctx.strokeStyle = (Math.abs(s - Math.round(s)) < 1e-9) ? "#e7d2d2" : "#f5e9e9";
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    ctx.strokeStyle = "#f5e9e9";
    for (let i = 0; i <= 6; i++) {
      const y = (i / 6) * h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
  }

  /**
   * Draw one trace.
   *   canvas    the <canvas class="ecg">
   *   values    the samples, raw ADC counts
   *   options   seconds  span of the x axis (default values.length / 100)
   *             leadOff  per-sample booleans; marked, never hidden
   *             ok       false greys the trace out (no valid ECG)
   */
  function draw(canvas, values, options) {
    options = options || {};
    const ctx = canvas.getContext("2d");
    const size = fitToDisplay(canvas);
    const seconds = options.seconds || Math.max(1, (values || []).length / 100);
    grid(ctx, size.w, size.h, seconds);

    if (!values || values.length < 2) {
      return;
    }

    // Autoscale. An ECG from the AD8232 occupies a few hundred of the 4096
    // ADC counts, so a fixed 0-4095 axis would flatten it into a straight line.
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < values.length; i++) {
      if (values[i] < lo) lo = values[i];
      if (values[i] > hi) hi = values[i];
    }
    const pad = Math.max(30, (hi - lo) * 0.15);
    lo -= pad;
    hi += pad;
    const span = (hi - lo) || 1;
    const y = v => size.h - ((v - lo) / span) * size.h;

    ctx.lineWidth = Math.max(1.2, 1.4 * size.ratio);
    ctx.strokeStyle = (options.ok === false) ? "#aab4bf" : "#1f5fa8";
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = (i / (values.length - 1)) * size.w;
      if (i === 0) ctx.moveTo(x, y(values[i])); else ctx.lineTo(x, y(values[i]));
    }
    ctx.stroke();

    // Lead-off samples get a tick along the top. They are not removed from the
    // trace: with a floating electrode the flags alternate, and masking them
    // would leave a trace with no two neighbouring points to join, which looks
    // like an empty screen rather than like a fault.
    const flags = options.leadOff;
    if (flags && flags.length) {
      ctx.strokeStyle = "#d9534f";
      ctx.lineWidth = Math.max(1.5, 2 * size.ratio);
      ctx.beginPath();
      for (let i = 0; i < flags.length && i < values.length; i++) {
        if (!flags[i]) continue;
        const x = (i / (values.length - 1)) * size.w;
        ctx.moveTo(x, 3 * size.ratio);
        ctx.lineTo(x, 13 * size.ratio);
      }
      ctx.stroke();
    }
  }

  return { draw: draw };
})();
