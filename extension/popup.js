const $ = (id) => document.getElementById(id);
const send = (msg) => new Promise((resolve) => chrome.runtime.sendMessage(msg, resolve));

function setStatus(text, cls = "") {
  $("status-text").textContent = text;
  $("status").className = cls;
}

function show(data) {
  setStatus(data.ready ? "Scoring server running" : "Server running, model not trained yet", data.ready ? "ok" : "");
  $("help").hidden = true;
  $("stats").hidden = false;
  for (const k of ["like", "dislike", "maybe"]) $(k).textContent = data.counts?.[k] ?? "–";
  $("cutoff").textContent = data.cutoff != null ? data.cutoff.toFixed(2) : "–";
  $("auc").textContent = data.auc != null ? data.auc.toFixed(2) : "–";
  $("retrain").disabled = false;
}

async function refresh() {
  const res = await send({ type: "status" });
  if (res?.ok) return show(res.data);
  setStatus("Scoring server isn’t running");
  $("help").innerHTML = "Start it in Terminal:<br><code>launchctl kickstart -k gui/$(id -u)/com.sarakong.tastescore</code>";
  $("help").hidden = false;
  $("stats").hidden = true;
  $("retrain").disabled = true;
}

chrome.storage.local.get({ enabled: true }, (v) => { $("enabled").checked = v.enabled; });
$("enabled").addEventListener("change", (e) => chrome.storage.local.set({ enabled: e.target.checked }));

$("retrain").addEventListener("click", async () => {
  $("retrain").disabled = true;
  setStatus("Retraining…", "ok busy");
  $("note").textContent = "";
  const res = await send({ type: "retrain" });
  if (!res?.ok) { $("note").textContent = res?.error || "Retraining failed."; await refresh(); return; }
  show(res.data);
  $("note").textContent = `Retrained in ${res.data.seconds}s · rescoring this page`;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id) chrome.tabs.sendMessage(tab.id, { type: "rescore" }).catch(() => {});
});

refresh();
