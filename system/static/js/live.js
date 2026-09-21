/*
 * live.js - the live ECG monitor.
 *
 * Both live pages show the same thing: a trace, the state of the link, the
 * state of the electrodes, and the last window the model scored. They differ
 * only in where the data comes from - MQTT over Wi-Fi, or a USB serial port -
 * and the two endpoints answer with different field names. So the page
 * supplies a poll function and a small adapter that returns a common shape,
 * and everything below is shared.
 *
 * Nothing here decides anything. The lead-off flag, the signal verdict, the
 * heart rate, the probability and the threshold are all computed on the
 * server; this file only puts them on screen.
 */

const LiveMonitor = (function () {
  "use strict";

  var $ = UI.$;

  function badge(id, text, kind, live) {
    var node = $(id);
    if (!node) return;
    node.className = "badge " + kind + (live ? " live" : "");
    node.innerHTML = '<span class="dot"></span>' + UI.escapeHtml(text);
  }

  function text(id, value) {
    var node = $(id);
    if (node) node.textContent = value;
  }

  /**
   * start(options)
   *   poll()        -> promise of the raw snapshot from the server
   *   adapt(raw)    -> the common view shape below
   *   intervalMs    -> polling period (250 ms on both pages today)
   *   onRaw(raw)    -> optional, for the page-specific bits (start/stop buttons)
   */
  function start(options) {
    var canvas = $("ecg");
    var lastResultAt = null;
    var lastTrace = [];
    var lastOk = true;
    var lastLead = null;
    var failures = 0;

    window.addEventListener("resize", function () {
      ECG.draw(canvas, lastTrace, { seconds: 10, leadOff: lastLead, ok: lastOk });
    });

    function render(v) {
      lastTrace = v.trace || [];
      lastOk = v.signalOk;
      lastLead = v.leadOffFlags;
      ECG.draw(canvas, lastTrace, {
        seconds: v.seconds || 10, leadOff: v.leadOffFlags, ok: v.signalOk
      });

      // The overlay states the one thing wrong with the signal, so nobody
      // reads a flat or noisy trace as an ECG. It is never a result. No link
      // and no data come first: a lead-off flag from a silent device is stale.
      var flag = $("ecgFlag");
      if (flag) {
        if (v.connected && v.online && v.leadOff) {
          flag.innerHTML = "<b>LEAD OFF</b><span>Attach the electrodes - waiting for a valid ECG signal</span>";
          flag.classList.remove("hidden");
        } else if (!v.connected) {
          flag.innerHTML = "<b>NO LINK</b><span>" +
              UI.escapeHtml(v.connectionText || "Device connection unavailable") + "</span>";
          flag.classList.remove("hidden");
        } else if (!v.online) {
          flag.innerHTML = "<b>NO DATA</b><span>Device is currently offline</span>";
          flag.classList.remove("hidden");
        } else if (!v.signalOk) {
          flag.innerHTML = "<b>NO VALID ECG</b><span>Waiting for a valid ECG signal</span>";
          flag.classList.remove("hidden");
        } else {
          flag.classList.add("hidden");
        }
      }

      text("diagnosis", v.diagnosis || "");

      badge("stLink", v.connectionText, v.connected ? "ok" : "flag", v.connected);
      badge("stDevice", v.online ? "RECEIVING" : "OFFLINE", v.online ? "ok" : "idle", v.online);
      showElectrodes(v);
      badge("stSignal", v.signalOk ? "VALID" : "NO VALID ECG", v.signalOk ? "ok" : "warn");

      text("bpm", v.bpm ? Math.round(v.bpm) : "—");
      text("rate", v.rateHz ? v.rateHz.toFixed(2) : "—");
      text("samples", v.samples || 0);
      text("counters", v.counters || "");
      text("window", (v.windowProgress || 0) + " / " + (v.windowLength || 1000));

      var bar = $("windowBar");
      if (bar) {
        var pct = 100 * (v.windowProgress || 0) / (v.windowLength || 1000);
        bar.style.width = Math.max(0, Math.min(100, pct)) + "%";
      }

      var discarded = Object.entries(v.windowsDiscarded || {})
        .filter(function (kv) { return kv[1]; })
        .map(function (kv) { return kv[0].replace(/_/g, " ") + " " + kv[1]; })
        .join(", ");
      text("windowsDone", "scored " + (v.windowsDone || 0) +
           (discarded ? " · discarded: " + discarded : ""));

      var r = v.lastResult;
      if (r && r.at !== lastResultAt) {
        lastResultAt = r.at;
        showResult(r, v.thresholds);
      }
    }

    // Electrode contact can only be judged from data that is arriving now.
    // With the device silent the answer is "unknown" - never "connected",
    // which is what the page used to say from a stale buffer.
    function showElectrodes(v) {
      var note = $("electrodeNote");
      if (!v.connected || !v.online) {
        badge("stLead", "UNKNOWN", "idle");
        badge("stLoPlus", "NO DATA", "idle");
        badge("stLoMinus", "NO DATA", "idle");
        if (note) note.textContent = "No data from the device, so electrode contact cannot be checked.";
        return;
      }
      var e = v.electrodes;          // per pin, when the transport reports it
      var plusOff = e ? e.lo_plus_off : v.leadOff;
      var minusOff = e ? e.lo_minus_off : v.leadOff;
      var anyOff = plusOff || minusOff;
      badge("stLead", anyOff ? "LEAD OFF" : "ATTACHED", anyOff ? "flag" : "ok");
      badge("stLoPlus", plusOff ? "OFF" : "ATTACHED", plusOff ? "flag" : "ok");
      badge("stLoMinus", minusOff ? "OFF" : "ATTACHED", minusOff ? "flag" : "ok");
      if (!note) return;
      if (!e && anyOff) {
        note.textContent = "An electrode is not making contact.";
      } else if (plusOff && minusOff) {
        note.textContent = "Both electrodes are off. Attach them and press them firmly onto the skin.";
      } else if (plusOff) {
        note.textContent = "The electrode on the +IN input (LO+) is not making contact.";
      } else if (minusOff) {
        note.textContent = "The electrode on the −IN input (LO−) is not making contact.";
      } else {
        note.textContent = "Both electrodes are making contact.";
      }
    }

    function showResult(r, thresholds) {
      var host = $("result");
      if (!host) return;
      if (r.error) {
        host.className = "notice err";
        host.innerHTML = UI.icon("i-alert") + "<div>" + UI.escapeHtml(r.error) + "</div>";
        return;
      }
      var mi = r.prediction === 1;
      var pct = (100 * r.probability).toFixed(1);
      var thrPct = (100 * r.threshold).toFixed(1);
      host.className = "result " + (mi ? "mi" : "normal");
      host.innerHTML =
        '<div class="headline">' +
          '<span class="prob">' + pct + '%</span>' +
          '<span class="badge ' + (mi ? "flag" : "ok") + '"><span class="dot"></span>' +
          (mi ? "MI PATTERN DETECTED" : "NORMAL PATTERN") + '</span>' +
        '</div>' +
        '<div class="meter ' + (mi ? "mi" : "") + '">' +
          '<div class="track">' +
            '<div class="fill" style="width:' + pct + '%"></div>' +
            '<div class="thr" style="left:' + thrPct + '%" title="threshold"></div>' +
          '</div>' +
          '<div class="ends"><span>0%</span>' +
          '<span>threshold ' + Number(r.threshold).toFixed(4) + ' (' +
          UI.escapeHtml(r.mode || "") + ')</span><span>100%</span></div>' +
        '</div>' +
        '<div class="meta">' +
          UI.escapeHtml(r.at || "") +
          (r.bpm ? " · " + Math.round(r.bpm) + " bpm" : "") +
          (r.inference_ms ? " · " + Math.round(r.inference_ms) + " ms" : "") +
          (r.report_id ? ' · <a href="/ecg/' + r.report_id + '">Open full report</a>' : "") +
        '</div>';
    }

    async function tick() {
      try {
        var raw = await options.poll();
        failures = 0;
        if (options.onRaw) options.onRaw(raw);
        render(options.adapt(raw));
      } catch (err) {
        failures++;
        // One lost poll on a hotspot is normal; several in a row is not, and
        // saying so once is better than a screen that silently freezes.
        if (failures === 4) {
          UI.toast(err.message || "Lost contact with the server.", "err");
          badge("stLink", "NO SERVER", "flag");
        }
      }
    }

    tick();
    setInterval(tick, options.intervalMs || 250);
    return { tick: tick };
  }

  return { start: start, badge: badge };
})();
