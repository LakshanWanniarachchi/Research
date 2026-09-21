/*
 * ecg.js - drawing an ECG trace on a canvas.
 *
 * One renderer for the live monitor and for a stored report. If each page
 * drew its own, a reader comparing a recording against the trace they
 * watched being taken would be comparing two renderers as much as two
 * signals.
 *
 * The panel is dark because a thin bright line holds its shape better
 * against dark than against white, which is why bedside monitors are built
 * that way. The grid is a reading aid at 0.2 s and 1 s; it is not calibrated
 * ECG paper, and the y axis is autoscaled raw ADC counts, so the page says so
 * rather than inviting millimetre measurements the signal cannot support.
 */

const ECG = (function () {
  "use strict";

  var INK_BG     = "#10202e";
  var GRID_MINOR = "rgba(255,255,255,.045)";
  var GRID_MAJOR = "rgba(255,255,255,.10)";
  var TRACE      = "#43d18c";
  var TRACE_DIM  = "#5b7488";
  var LEAD_OFF   = "#f4795b";
  var AXIS_TEXT  = "rgba(230,238,245,.45)";

  function fitToDisplay(canvas) {
    // Sized in CSS, drawn in device pixels. Without this the trace is blurred
    // on a high-DPI screen, and the thing being read is the shape of a QRS
    // complex.
    var ratio = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    var w = Math.max(1, Math.round(rect.width * ratio));
    var h = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    return { w: canvas.width, h: canvas.height, ratio: ratio };
  }

  function grid(ctx, size, seconds) {
    var w = size.w, h = size.h, s, x, i, y;
    ctx.fillStyle = INK_BG;
    ctx.fillRect(0, 0, w, h);

    ctx.lineWidth = Math.max(1, size.ratio * 0.6);
    for (s = 0; s <= seconds + 1e-9; s += 0.2) {
      x = (s / seconds) * w;
      ctx.strokeStyle = (Math.abs(s - Math.round(s)) < 1e-9) ? GRID_MAJOR : GRID_MINOR;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    ctx.strokeStyle = GRID_MINOR;
    for (i = 0; i <= 8; i++) {
      y = (i / 8) * h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Seconds along the bottom. The x axis is the one axis that is true.
    ctx.fillStyle = AXIS_TEXT;
    ctx.font = (10 * size.ratio) + "px system-ui, sans-serif";
    ctx.textBaseline = "bottom";
    for (s = 0; s <= seconds + 1e-9; s += 1) {
      if (s === 0) continue;
      x = (s / seconds) * w;
      ctx.textAlign = (s >= seconds - 1e-9) ? "right" : "center";
      ctx.fillText(s + "s", x, h - 3 * size.ratio);
    }
  }

  /**
   * Draw one trace.
   *   canvas   the <canvas class="ecg">
   *   values   samples, raw ADC counts
   *   options  seconds  span of the x axis (default values.length / 100)
   *            leadOff  per-sample booleans; marked, never hidden
   *            ok       false greys the trace (no valid ECG)
   */
  function draw(canvas, values, options) {
    options = options || {};
    var ctx = canvas.getContext("2d");
    var size = fitToDisplay(canvas);
    var seconds = options.seconds || Math.max(1, (values || []).length / 100);
    grid(ctx, size, seconds);

    if (!values || values.length < 2) return;

    // Autoscale: an AD8232 ECG occupies a few hundred of the 4096 ADC counts,
    // so a fixed 0-4095 axis would flatten it into a straight line.
    var lo = Infinity, hi = -Infinity, i, v;
    for (i = 0; i < values.length; i++) {
      v = values[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    var pad = Math.max(30, (hi - lo) * 0.15);
    lo -= pad; hi += pad;
    var span = (hi - lo) || 1;
    var y = function (val) { return size.h - ((val - lo) / span) * size.h; };

    ctx.lineWidth = Math.max(1.3, 1.5 * size.ratio);
    ctx.lineJoin = "round";
    ctx.strokeStyle = (options.ok === false) ? TRACE_DIM : TRACE;
    if (options.ok !== false) {
      ctx.shadowColor = "rgba(67, 209, 140, .35)";
      ctx.shadowBlur = 4 * size.ratio;
    }
    ctx.beginPath();
    for (i = 0; i < values.length; i++) {
      var x = (i / (values.length - 1)) * size.w;
      if (i === 0) ctx.moveTo(x, y(values[i])); else ctx.lineTo(x, y(values[i]));
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Lead-off samples get a tick along the top. They are not removed from
    // the trace: with a floating electrode the flags alternate, and masking
    // them would leave no two neighbouring points to join, which looks like
    // an empty screen rather than like a fault.
    var flags = options.leadOff;
    if (flags && flags.length) {
      ctx.strokeStyle = LEAD_OFF;
      ctx.lineWidth = Math.max(1.5, 2 * size.ratio);
      ctx.beginPath();
      for (i = 0; i < flags.length && i < values.length; i++) {
        if (!flags[i]) continue;
        var fx = (i / (values.length - 1)) * size.w;
        ctx.moveTo(fx, 3 * size.ratio);
        ctx.lineTo(fx, 13 * size.ratio);
      }
      ctx.stroke();
    }
  }

  /** Redraw on resize without the caller keeping its own copy of the data. */
  function bind(canvas, getValues, getOptions) {
    var redraw = function () {
      draw(canvas, getValues(), getOptions ? getOptions() : {});
    };
    window.addEventListener("resize", redraw);
    redraw();
    return redraw;
  }

  return { draw: draw, bind: bind };
})();
