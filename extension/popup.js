const $ = (id) => document.getElementById(id);
const send = (msg) => new Promise((resolve) => chrome.runtime.sendMessage(msg, resolve));

function show(data) {
  $("status").textContent = data.ready ? "Scoring server running" : "Server running, model not trained yet";
  $("status").className = data.ready ? "ok" : "";
  $("stats").hidden = false;
  for (const k of ["like", "dislike", "maybe"]) $(k).textContent = data.counts?.[k] ?? "–";
  $("cutoff").textContent = data.cutoff != null ? data.cutoff.toFixed(2) : "–";
  $("auc").textContent = data.auc != null ? data.auc.toFixed(2) : "–";
  $("retrain").disabled = false;
}

async function refresh() {
  const res = await send({ type: "status" });
  if (res?.ok) return show(res.data);
  $("status").innerHTML = "Scoring server isn't running. In Terminal, run <code>python server.py</code> in your style-matcher folder.";
  $("retrain").disabled = true;
}

chrome.storage.local.get({ enabled: true }, (v) => { $("enabled").checked = v.enabled; });
$("enabled").addEventListener("change", (e) => chrome.storage.local.set({ enabled: e.target.checked }));

$("retrain").addEventListener("click", async () => {
  $("retrain").disabled = true;
  $("note").textContent = "Retraining…";
  const res = await send({ type: "retrain" });
  if (!res?.ok) { $("note").textContent = res?.error || "Retraining failed."; $("retrain").disabled = false; return; }
  show(res.data);
  $("note").textContent = `Retrained in ${res.data.seconds}s. Rescoring this page.`;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id) chrome.tabs.sendMessage(tab.id, { type: "rescore" }).catch(() => {});
});

refresh();
