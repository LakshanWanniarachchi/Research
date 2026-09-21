/*
 * ui.js - the small pieces of behaviour every page shares.
 *
 * The navigation drawer, toasts, the password field, and one place where a
 * failed request turns into a message a person can read. Nothing here knows
 * anything about ECGs; that is in ecg.js and live.js.
 */

const UI = (function () {
  "use strict";

  function $(id) { return document.getElementById(id); }
  function el(sel, root) { return (root || document).querySelector(sel); }
  function els(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  function icon(name, cls) {
    return '<svg class="ico ' + (cls || "") + '" width="16" height="16" aria-hidden="true">' +
           '<use href="/static/icons.svg#' + name + '"></use></svg>';
  }

  // --- toasts ------------------------------------------------------------
  // One notification path for the whole application. A page that wants to
  // report something calls UI.toast; nothing writes its own floating box.
  var ICONS = { ok: "i-check", warn: "i-warn", err: "i-alert", info: "i-info" };

  function toast(message, kind, ms) {
    kind = kind || "info";
    var host = $("toasts");
    if (!host) return;
    var node = document.createElement("div");
    node.className = "toast " + kind;
    node.setAttribute("role", kind === "err" ? "alert" : "status");
    node.innerHTML = icon(ICONS[kind] || ICONS.info) +
                     '<div>' + escapeHtml(message) + '</div>' +
                     '<button type="button" aria-label="Dismiss">' + icon("i-x") + '</button>';
    node.querySelector("button").onclick = function () { node.remove(); };
    host.appendChild(node);
    setTimeout(function () { node.remove(); }, ms || 5000);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // --- fetch -------------------------------------------------------------
  // Every API call in the application goes through here, so a server error
  // reaches the user as a sentence rather than as a stack trace or as a page
  // that quietly stops updating.
  async function getJSON(url) {
    var resp = await fetch(url, { headers: { "Accept": "application/json" } });
    if (resp.status === 401 || resp.status === 403) {
      throw new Error("Your session has ended. Please sign in again.");
    }
    if (resp.status === 404) {
      var e404 = new Error("Not found.");
      e404.status = 404;
      throw e404;
    }
    if (!resp.ok) {
      throw new Error("Something went wrong on the server (" + resp.status + ").");
    }
    return resp.json();
  }

  async function postJSON(url, body) {
    var resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(body || {})
    });
    var data = {};
    try { data = await resp.json(); } catch (e) { /* empty body */ }
    if (!resp.ok && !data.error) data.error = "Request failed (" + resp.status + ").";
    return data;
  }

  // --- navigation drawer -------------------------------------------------
  function initNav() {
    var app = el(".app");
    var btn = $("menuBtn");
    var scrim = el(".scrim");
    if (!app || !btn) return;

    function setOpen(open) {
      app.classList.toggle("nav-open", open);
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    }
    btn.onclick = function () { setOpen(!app.classList.contains("nav-open")); };
    if (scrim) scrim.onclick = function () { setOpen(false); };
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") setOpen(false);
    });
    // A link tapped in the drawer should close it, or the new page appears
    // behind an open menu.
    els(".side-nav a").forEach(function (a) {
      a.addEventListener("click", function () { setOpen(false); });
    });
  }

  // --- password fields ---------------------------------------------------
  function initPasswords() {
    els("[data-pw]").forEach(function (input) {
      var wrap = document.createElement("div");
      wrap.className = "pw-wrap";
      input.parentNode.insertBefore(wrap, input);
      wrap.appendChild(input);

      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pw-toggle";
      btn.setAttribute("aria-label", "Show password");
      btn.innerHTML = icon("i-eye");
      btn.onclick = function () {
        var shown = input.type === "text";
        input.type = shown ? "password" : "text";
        btn.innerHTML = icon(shown ? "i-eye" : "i-eye-off");
        btn.setAttribute("aria-label", shown ? "Show password" : "Hide password");
      };
      wrap.appendChild(btn);
    });

    var meterFor = el("[data-pw-meter]");
    if (meterFor) {
      var target = $(meterFor.getAttribute("data-pw-meter"));
      if (target) {
        target.addEventListener("input", function () {
          meterFor.className = "pw-meter s" + strength(target.value);
        });
      }
    }
  }

  // Four bands, not a score out of a hundred: length first, because it is
  // what actually matters, then variety. The server still enforces the rule;
  // this only tells the person where they stand while typing.
  function strength(pw) {
    if (!pw) return 0;
    var score = 0;
    if (pw.length >= 8) score++;
    if (pw.length >= 12) score++;
    if (/[a-z]/.test(pw) && /[A-Z]/.test(pw)) score++;
    if (/[0-9]/.test(pw) && /[^A-Za-z0-9]/.test(pw)) score++;
    return Math.max(1, Math.min(4, score));
  }

  // --- time --------------------------------------------------------------
  function ago(iso) {
    if (!iso) return "never";
    var then = Date.parse(iso.replace(" ", "T"));
    if (isNaN(then)) return iso;
    var s = Math.max(0, Math.round((Date.now() - then) / 1000));
    if (s < 5) return "just now";
    if (s < 60) return s + " s ago";
    if (s < 3600) return Math.round(s / 60) + " min ago";
    if (s < 86400) return Math.round(s / 3600) + " h ago";
    return Math.round(s / 86400) + " d ago";
  }

  document.addEventListener("DOMContentLoaded", function () {
    initNav();
    initPasswords();
  });

  return {
    $: $, el: el, els: els, icon: icon, toast: toast, escapeHtml: escapeHtml,
    getJSON: getJSON, postJSON: postJSON, ago: ago, strength: strength
  };
})();
